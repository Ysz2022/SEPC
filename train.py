import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import utils
import torch.nn.functional as F
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.image_io import save_graph
from model.SEPC import SEPC


parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,60,90], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument("--save_path", type=str, default="logs/SEPC_derainH", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=3,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="trainData/RainTrainH",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
opt = parser.parse_args()

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def main():
    os.makedirs(opt.save_path, exist_ok=True)
    losses = []
    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    num_train = dataset_train.__len__()
    indices = list(range(num_train))
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size,
                              sampler=torch.utils.data.sampler.SequentialSampler(indices[:]),pin_memory=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = SEPC(3,3)

    # loss function
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        mse.cuda()
        l1.cuda()

    # model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch46.pth')))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # start training
    for epoch in range(0,opt.epochs):
        scheduler.step(epoch)
        print('epoch:', epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        objs = utils.AvgrageMeter()
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)
            n = input_train.size(0)
            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            final_out, out_train, x1_1, x2_1 = model(input_train)

            loss = 200 * l1(target_train, out_train) / (l1(x1_1, x2_1) + 100 * l1(input_train, out_train))
            loss2 = 2 * l1(target_train, final_out) / l1(input_train, final_out)

            loss += loss2

            loss.backward()
            optimizer.step()

            objs.update(loss.data.item(), n)

        print(objs.avg)
        ## epoch training end


        # # save model
        # torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
            # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))

        losses.append(objs.avg)
        save_graph(str(epoch) + "_aloss", losses,
                   output_path=opt.save_path + '/')


if __name__ == "__main__":
    # prepare_data_RainTrainL(data_path=opt.data_path, patch_size=128, stride=80)
    prepare_data_RainTrainH(data_path=opt.data_path, patch_size=128, stride=80)
    main()
