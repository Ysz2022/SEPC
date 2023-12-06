import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
# from unet.unet_cab12_new import UNet
from model.SEPC import SEPC
import time


parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--logdir", type=str, default="logs/SEPC_derainL/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="testData/rain100/blended", help='path to training data')
parser.add_argument("--save_path", type=str, default="results/100", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# if opt.use_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def normalize(data):
    return data / 255.

def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]

def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = SEPC(3,3)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_epoch106.pth')))
    # # 重新保存网络参数，此时注意改为非zip格式
    # torch.save(model.state_dict(), os.path.join(opt.logdir, 'final_vgg.pth'), _use_new_zipfile_serialization=False)

    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        img_path = os.path.join(opt.data_path, img_name)

        # input image
        y = cv2.imread(img_path)
        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])

        y = normalize(np.float32(y))
        y = np.expand_dims(y.transpose(2, 0, 1), 0)
        y = crop_torch_image(torch.Tensor(y), d=32)
        y = Variable(y)

        if opt.use_GPU:
            y = y.cuda()

        with torch.no_grad(): #
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()

            # print('y: ', y.size())
            final,out0,_,_ = model(y)
            out = torch.clamp(final, 0., 1.)

            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time

            print(img_name, ': ', dur_time)

        if opt.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())

        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        our_name = img_name.split('.')[0]
        our_name = our_name + '.png'
        cv2.imwrite(os.path.join(opt.save_path, our_name), save_out)

        count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

