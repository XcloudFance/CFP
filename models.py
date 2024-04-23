from functools import partial
from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F

from CFPFormer import CFPNet


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlockBN(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockBN, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)





class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

from torchvision.models import resnet50

class UNet16Resnet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.pool = nn.MaxPool2d(2, 2)

        self.adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


        print('pretrained')
        self.encoder = resnet50(pretrained=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.CFPNet = CFPNet(
            embed_dims=[2048, 1024, 512, 256, 64],
            in_chans=2048,  # Update according to the output channels of your ResNet backbone
            #embed_dims=[512, 512, 256, 128, 64],
            depths=[1, 2, 3, 2, 2],
            num_heads=[2, 4, 8, 16, 32],
            init_values=[2, 2, 2, 2,2],
            heads_ranges=[4, 4, 6, 6,6],
            mlp_ratios=[2, 2, 2, 2,2],
            drop_path_rate=0.1,
            chunkwise_recurrents=[True, True, True, True, True],
            layerscales=[False, False, False, False, False]
        )
        self.center = DecoderBlock(2048, num_filters * 8 * 2, 2048)
        self.dec1 = DecoderBlock(320, 128 , 192, True)
        self.dec_final = DecoderBlock(192,128,64,True)
        self.SegmentationHead = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        print(1, x.shape)
        x = self.adapter(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        #print(2, conv5.shape)
        center = self.center(self.pool(conv5))
        
        encode_features = [conv5, conv4, conv3, conv2, conv1]

        #print(center.shape, conv4.shape,conv3.shape,conv2.shape,conv1.shape)
        x = self.CFPNet(center, encode_features)
        
        x = x.permute(0, 3, 1, 2)
        x = self.dec1(torch.cat([x,conv2],1))
        x = self.dec_final(x)
        x = self.SegmentationHead(x)
        return x

