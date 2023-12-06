""" Full assembly of the parts to form the complete network """
from model.unet_parts import *

class SEPC(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SEPC, self).__init__()
        act = nn.PReLU()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shallow_feat = nn.Sequential(conv(n_channels, 64, 3, bias=False),
                                           CAB(64, 3, 16, bias=False, act=act))

        self.conv1 = conv(in_channels=64, out_channels=64, kernel_size=1, bias=False)
        self.conv3 = conv(3, 64, 1, bias=False)

        self.inc = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self.shallow_feat3 = nn.Sequential(conv(3, 64, 3), CAB(64, 3, 16, bias=False, act=act))
        self.stage3_orsnet = ORSNet(n_feat=64, scale_orsnetfeats=32, kernel_size=3, reduction=16, act=act, bias=False,
                                    scale_unetfeats=48, num_cab=12)
        self.concat23 = conv(64 * 2, 96, 3)
        self.tail = conv(96, 3, 1)

    def forward(self, x):
        x1_1 = self.shallow_feat(x)
        x1_1 = self.inc(x1_1)

        x1_2 = self.down1(x1_1)
        x1_3 = self.down2(x1_2)
        x1_4 = self.down3(x1_3)
        x5 = self.down4(x1_4)
        x2_4 = self.up1(x5, x1_4)
        x2_3 = self.up2(x2_4, x1_3)
        x2_2 = self.up3(x2_3, x1_2)
        x2_1 = self.up4(x2_2, x1_1)

        x3_samfeats, img = self.SAM(x2_1, x)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x)
        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat, x1_1, x2_1)
        stage3_img = self.tail(x3_cat)


        return stage3_img + x, img, x1_1, x2_1


    def SAM(self, x, x_img):
        x1 = self.conv1(x)
        img = self.outc(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img





if __name__ == '__main__':
    a = torch.randn(4, 3, 256, 256)
    mode = UNet(3,3)
    b,_,_,_ = mode(a)
    b = a[:, 0, :, :]
    print(b.size())