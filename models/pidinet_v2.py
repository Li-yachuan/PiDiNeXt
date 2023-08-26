import torch
from torch import nn
from .ops import createConvFunc
from .pidinet import Conv2d, CSAM, CDCM, MapReduce
import torch.nn.functional as F
from .mnn import *
from collections import OrderedDict


# Parallel pixal differential module
class PPDM(nn.Module):
    def __init__(self, ouplane):
        super(PPDM, self).__init__()
        self.init_block_cv = Conv2d(createConvFunc('cv'), 3, ouplane, kernel_size=3, padding=1)
        self.init_block_cd = Conv2d(createConvFunc('cd'), 3, ouplane, kernel_size=3, padding=1)
        self.init_block_ad = Conv2d(createConvFunc('ad'), 3, ouplane, kernel_size=3, padding=1)
        self.init_block_rd = Conv2d(createConvFunc('rd'), 3, ouplane, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.init_block_ad(x) + self.init_block_cd(x) + self.init_block_cv(x) + self.init_block_rd(x)) / 4
        return self.relu(x)


class PPDM_convert(nn.Module):
    def __init__(self, ouplane):
        super(PPDM_convert, self).__init__()
        self.init_block = nn.Conv2d(3, ouplane, kernel_size=(5, 5), padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.init_block(x)
        return self.relu(x)


class PDCBlock(nn.Module):
    def __init__(self, inplane, ouplane, stride=1, bias=False, act=None):
        super(PDCBlock, self).__init__()

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        else:
            self.pool = nn.Identity()
            self.shortcut = nn.Identity()

        self.conv_cv = Conv2d(createConvFunc('cv'), inplane, inplane, kernel_size=3, padding=1, groups=inplane,
                              bias=bias)
        self.conv_cd = Conv2d(createConvFunc('cd'), inplane, inplane, kernel_size=3, padding=1, groups=inplane,
                              bias=bias)
        self.conv_ad = Conv2d(createConvFunc('ad'), inplane, inplane, kernel_size=3, padding=1, groups=inplane,
                              bias=bias)
        self.conv_rd = Conv2d(createConvFunc('rd'), inplane, inplane, kernel_size=3, padding=1, groups=inplane,
                              bias=bias)

        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.relu2 = act()

    def forward(self, x):

        x = self.pool(x)
        y = (self.conv_cv(x) + self.conv_cd(x) + self.conv_ad(x) + self.conv_rd(x)) / 4
        y = self.conv2(self.relu2(y))
        y = y + self.shortcut(x)
        return y


class PDCBlock_convert(nn.Module):
    def __init__(self, inplane, ouplane, stride=1, bias=False, act=None):
        super(PDCBlock_convert, self).__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=(1, 1), padding=0)
        else:
            self.pool = nn.Identity()
            self.shortcut = nn.Identity()

        self.conv = nn.Conv2d(inplane, inplane, kernel_size=(5, 5), padding=2, groups=inplane, bias=bias)
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=(1, 1), padding=0)
        self.relu2 = act(inplace=True)

    def forward(self, x):

        x = self.pool(x)
        y = self.conv(x)
        y = self.conv2(self.relu2(y))
        y = y + self.shortcut(x)
        return y

class PiDiNetv2(nn.Module):
    def __init__(self, inplane, dil=None, sa=False, convert=False, act=None):
        super(PiDiNetv2, self).__init__()
        self.sa = sa
        act = eval(act)

        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane

        if convert:
            self.init_block = PPDM_convert(self.inplane)
            block_class = PDCBlock_convert

        else:
            self.init_block = PPDM(self.inplane)
            block_class = PDCBlock

        self.block1_1 = block_class(self.inplane, self.inplane, act=act)
        self.block1_2 = block_class(self.inplane, self.inplane, act=act)
        self.block1_3 = block_class(self.inplane, self.inplane, act=act)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(inplane, self.inplane, stride=2, act=act)
        self.block2_2 = block_class(self.inplane, self.inplane, act=act)
        self.block2_3 = block_class(self.inplane, self.inplane, act=act)
        self.block2_4 = block_class(self.inplane, self.inplane, act=act)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(inplane, self.inplane, stride=2, act=act)
        self.block3_2 = block_class(self.inplane, self.inplane, act=act)
        self.block3_3 = block_class(self.inplane, self.inplane, act=act)
        self.block3_4 = block_class(self.inplane, self.inplane, act=act)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(self.inplane, self.inplane, stride=2, act=act)
        self.block4_2 = block_class(self.inplane, self.inplane, act=act)
        self.block4_3 = block_class(self.inplane, self.inplane, act=act)
        self.block4_4 = block_class(self.inplane, self.inplane, act=act)
        self.fuseplanes.append(self.inplane)  # 4C

        self.predict_heads = nn.ModuleList()

        if self.sa and self.dil is not None:
            for i in range(4):
                self.predict_heads.append(
                    nn.Sequential(CDCM(self.fuseplanes[i], self.dil),
                                  CSAM(self.dil),
                                  MapReduce(self.dil)))
        elif self.sa:
            for i in range(4):
                self.predict_heads.append(
                    nn.Sequential(CSAM(self.fuseplanes[i]),
                                  MapReduce(self.fuseplanes[i])))
        elif self.dil is not None:
            for i in range(4):
                self.predict_heads.append(
                    nn.Sequential(CDCM(self.fuseplanes[i], self.dil),
                                  MapReduce(self.dil)))
        else:
            for i in range(4):
                self.predict_heads.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=(1, 1))  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]
        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        outputs = []
        for i, xi in enumerate([x1, x2, x3, x4]):
            outputs.append(
                F.interpolate(self.predict_heads[i](xi), (H, W), mode="bilinear", align_corners=False))

        outputs.append(self.classifier(torch.cat(outputs, dim=1)))

        return [torch.sigmoid(r) for r in outputs]


def pidinet_tiny_v2(args):
    dil = 8 if args.dil else None
    return PiDiNetv2(20, dil=dil, sa=args.sa, act=args.act)


def pidinet_small_v2(args):
    dil = 12 if args.dil else None
    return PiDiNetv2(30, dil=dil, sa=args.sa, act=args.act)


def pidinet_v2(args):
    dil = 24 if args.dil else None
    return PiDiNetv2(60, dil=dil, sa=args.sa, act=args.act)

def pidinet_large_v2(args):
    dil = 24 if args.dil else None
    return PiDiNetv2(120, dil=dil, sa=args.sa, act=args.act)


def pidinet_tiny_v2_converted(args):
    dil = 8 if args.dil else None
    return PiDiNetv2(20, dil=dil, sa=args.sa, convert=True, act=args.act)


def pidinet_small_v2_converted(args):
    dil = 12 if args.dil else None
    return PiDiNetv2(30, dil=dil, sa=args.sa, convert=True, act=args.act)


def pidinet_v2_converted(args):
    dil = 24 if args.dil else None
    return PiDiNetv2(60, dil=dil, sa=args.sa, convert=True, act=args.act)
