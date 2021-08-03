import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import importlib
from models.resnet import resnet18,resnet50,resnet34,resnet101
from layers.functions import PriorBox
from layers.functions.prior_layer import PriorLayer
from data import VOC_300,COCO_300
from dcn.modules.deform_conv import DeformConv, ModulatedDeformConv
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#RFAM
class RFAM(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFAM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


#RFAM_PRO
class RFAM_PRO(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(RFAM_PRO, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


def add_dcn_dilas():
    planes = [512, 1024, 512, 256, 256, 256]
    deformable_groups = 1
    conv_layers = []
    for i in range(6):
        conv_layers += [DeformConv(
            planes[i],
            256,
            kernel_size=3,
            stride=1,
            padding=5 - i,
            dilation=5 - i,
            deformable_groups=deformable_groups,
            bias=False)]
    return conv_layers

def BN_layers():
    bn_layers = []
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    bn_layers += [nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)]
    return bn_layers

class FERNet(nn.Module):
    """FERNet for Underwater object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,cfg):
        super(FERNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        #refine
        self.odm_channels = [256, 256, 256, 256, 256, 256]
        self.input_fixed = True

        self.arm_num_classes = 2
        self.odm_loc = nn.ModuleList()
        self.odm_conf = nn.ModuleList()
        self.loc_offset_conv = nn.ModuleList()
        self.dcn_convs = nn.ModuleList(add_dcn_dilas())
        self.bn_layers = nn.ModuleList(BN_layers())
        self.prior_layer = PriorLayer(cfg)
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()


        # vgg network
        self.base = nn.ModuleList(base)
        self.assistant = resnet50(pretrained=True)
        # conv_4
        self.Norm = RFAM_PRO(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.arm_channels = [512, 1024, 512, 256, 256, 256]

        self.num_anchors = [6, 6, 6, 6, 4, 4]


        self.arm_loc = nn.ModuleList()
        self.arm_conf = nn.ModuleList()


        #cascade backbone
        self.cutchann = nn.Sequential(nn.Conv2d(192, 128, (1, 1), 1, padding=0))
        self.cutchanne = nn.Sequential(nn.Conv2d(512, 256, (1, 1), 1, padding=0))
        self.cutchannel = nn.Sequential(nn.Conv2d(1024, 512, (1, 1), 1, padding=0))
        self.cutchannell = nn.Sequential(nn.Conv2d(2048, 1024, (1, 1), 1, padding=0))


        for i in range(len(self.arm_channels)):

            self.arm_loc += [
                nn.Conv2d(
                    self.arm_channels[i],
                    self.num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1)
            ]
            self.arm_conf += [
                nn.Conv2d(
                    self.arm_channels[i],
                    self.num_anchors[i] * self.arm_num_classes,
                    kernel_size=3,
                    padding=1)
            ]

            self.loc_offset_conv +=[BasicConv(self.num_anchors[i] * 2, 18, kernel_size=1)]
            self.odm_loc += [
                nn.Conv2d(
                    self.odm_channels[i],
                    self.num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1)
            ]
            self.odm_conf += [
                nn.Conv2d(
                    self.odm_channels[i],
                    self.num_anchors[i] * self.num_classes,
                    kernel_size=3,
                    padding=1)
            ]
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test(focalloss&crossentropy):
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        arm_sources = list()
        arm_loc = list()
        arm_loc_convcg = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        odm_xs_n = list()
        arm_loc_list = list()
        conf = list()


        net1 = self.assistant(x)
        m, x0, x1, x2, x3, x4 = net1


        # apply vgg up to conv4_3 relu


        # add bn vgg16
        # for k in range(32):
        #     x = self.base[k](x)
        # s = self.Norm(x)
        # arm_sources.append(s)

        # for k in range(32, len(self.base)):
        #     x = self.base[k](x)


        #CASCADE BACKBONE
        for k in range(9):
            x = self.base[k](x)

        x = torch.cat((x, x0), 1)
        x = self.cutchann(x)

        for k in range(9, 16):
            x = self.base[k](x)

        x = torch.cat((x, x1), 1)
        x = self.cutchanne(x)
        for k in range(16, 23):
            x = self.base[k](x)

        x = torch.cat((x, x2), 1)
        x = self.cutchannel(x)

        s = self.Norm(x)
        arm_sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                arm_sources.append(x)

        odm_sources = arm_sources
        arm_xs = arm_sources
        odm_xs = odm_sources

        # apply multibox head to source layers
        for (x, l, c) in zip(arm_xs, self.arm_loc, self.arm_conf):

            arm_loc_conv = l(x)
            arm_loc_convcg.append(arm_loc_conv)
            cc = c(x)
            conf.append(cc)
            arm_loc_list.append(torch.cat([arm_loc_conv[:, 0::4, :, :], arm_loc_conv[:, 1::4, :, :]], 1))
            arm_loc.append(arm_loc_conv.permute(0, 2, 3, 1).contiguous())
            arm_conf.append(cc.permute(0, 2, 3, 1).contiguous())


        for (conf_fea, odm_xs_fea) in zip(conf, odm_xs):
            conf_obj = conf_fea[:, 1::2, :, :]
            conf_max, _ = torch.max(conf_obj, dim=1, keepdim=True)
            conf_attention = conf_max.sigmoid()
            odm_xs_fea_n = odm_xs_fea * conf_attention + odm_xs_fea
            odm_xs_n.append(odm_xs_fea_n)



        offset_0 = self.loc_offset_conv[0](arm_loc_list[0])
        d0 = F.relu(self.bn_layers[0](self.dcn_convs[0](odm_xs_n[0], offset_0)), inplace=True)

        offset_1 = self.loc_offset_conv[1](arm_loc_list[1])
        d1 = F.relu(self.bn_layers[1](self.dcn_convs[1](odm_xs_n[1], offset_1)), inplace=True)

        offset_2 = self.loc_offset_conv[2](arm_loc_list[2])
        d2 = F.relu(self.bn_layers[2](self.dcn_convs[2](odm_xs_n[2], offset_2)), inplace=True)


        offset_3 = self.loc_offset_conv[3](arm_loc_list[3])
        d3 = F.relu(self.bn_layers[3](self.dcn_convs[3](odm_xs_n[3], offset_3)), inplace=True)


        offset_4 = self.loc_offset_conv[4](arm_loc_list[4])
        d4 = F.relu(self.bn_layers[4](self.dcn_convs[4](odm_xs_n[4], offset_4)), inplace=True)

        d5 = arm_xs[5]

        odm_xs_new = [d0, d1, d2, d3, d4, d5]

        for (x, l, c) in zip(odm_xs_new, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        # print([o.size() for o in loc])

        img_wh = (x.size(3), x.size(2))
        feature_maps_wh = [(t.size(3), t.size(2)) for t in arm_xs]

        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)


        output = (
            arm_loc.view(arm_loc.size(0), -1, 4),
            arm_conf.view(arm_conf.size(0), -1, self.arm_num_classes),
            odm_loc.view(odm_loc.size(0), -1, 4),
            odm_conf.view(odm_conf.size(0), -1, self.num_classes),
            self.priors if self.input_fixed else self.prior_layer(img_wh, feature_maps_wh)
        )
        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


#add batchnorm vgg16
# def vgg(cfg, i, batch_norm=False, pool5_ds=False):
#     layers = []
#     in_channels = i
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=False)]
#             in_channels = v
#     if pool5_ds:
#         pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#     else:
#         pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
#     conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
#     # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     if batch_norm:
#         layers += [pool5, conv6, nn.BatchNorm2d(conv6.out_channels),
#                    nn.ReLU(inplace=False), conv7, nn.BatchNorm2d(conv7.out_channels), nn.ReLU(inplace=False)]
#     else:
#         layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
#     return layers


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [RFAM(in_channels, cfg[k+1], stride=2, scale=1.0, visual=1)]
                else:
                    layers += [RFAM(in_channels, cfg[k+1], stride=2, scale=1.0, visual=2)]
            else:
                layers += [RFAM(in_channels, v, scale=1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
    else:
        print("Error: Sorry only FERNet300 and FERNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only FERNet300 and FERNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, cfg, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only FERNet300 and FERNet512 are supported!")
        return

    config = cfg

    return FERNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes, config)