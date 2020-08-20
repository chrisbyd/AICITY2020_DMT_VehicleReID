import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
BG_DIM = 16
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm, activation, pad_type):
        super(Conv2dBlock, self).__init__()

        self.use_bias = True

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation, pad_type):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ImgDecoder(nn.Module):
    def __init__(self, cfg, feature_size=2048, bg_dim=BG_DIM):
        super(ImgDecoder, self).__init__()
        self.decoder_layer = []
        dim = feature_size + bg_dim
        self.decoder_layer += [ResBlocks(1, dim, "bn", "relu", pad_type="reflect"), nn.Upsample(scale_factor=4),
                               Conv2dBlock(dim, 32, 5, 1, 2, norm='bn', activation="relu", pad_type="reflect"),
                               nn.Upsample(scale_factor=4),
                               Conv2dBlock(32, 3, 5, 1, 2, norm='none', activation="tanh", pad_type="reflect")
                               ]
        self.decoder_layer = nn.Sequential(*self.decoder_layer)

    def forward(self, disentangled_feature_map, project_feature, bg_feature):
        assert len(project_feature.shape) == 2
        assert disentangled_feature_map.shape[0] == project_feature.shape[0]
        assert disentangled_feature_map.shape[0] == bg_feature.shape[0]
        projected_feature = disentangled_feature_map * project_feature[:, :, None, None]
        img_feature = torch.cat([projected_feature, bg_feature], dim=1)

        img = self.decoder_layer(img_feature)

        return img


class CutterNet(nn.Module):
    def __init__(self, cfg):
        super(CutterNet, self).__init__()
        dim = BG_DIM
        self.mask_layer = [ResBlocks(1, BG_DIM, "bn", "relu", pad_type="reflect"), nn.Upsample(scale_factor=4),
                               Conv2dBlock(dim, 8, 5, 1, 2, norm='bn', activation="relu", pad_type="reflect"),
                               nn.Upsample(scale_factor=4),
                               Conv2dBlock(8, 1, 5, 1, 2, norm='none', activation="tanh", pad_type="reflect")
                               ]
        self.mask_layer = nn.Sequential(*self.mask_layer)

    def forward(self, bg_feature):
        cut = (self.mask_layer(bg_feature) + 1) / 2
        cut = cut.repeat(1, 3, 1, 1)
        return cut

class TakerNet(nn.Module):
    def __init__(self, cfg, clss_num):
        super(TakerNet, self).__init__()
        self.model = []
        dim = 8
        norm = "bn"
        activ = "relu"
        pad_type = "reflect"
        style_dim = 64
        self.model += [Conv2dBlock(3, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [nn.AdaptiveAvgPool2d(4)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

        self.classifier = nn.Linear(style_dim, clss_num)

    def forward(self, x):
        student_code = self.model(x)
        student_code = nn.functional.avg_pool2d(student_code, student_code.shape[2:4])
        student_code = student_code.view(student_code.shape[0], -1)
        predict = self.classifier(student_code)
        print("taker code", student_code.shape, predict.shape)

        return student_code, predict

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg, bg_dim=BG_DIM):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            try:
                self.base.load_param(model_path)
            except:
                state_dict = load_state_dict_from_url(model_urls[model_name], model_dir=model_path)
                self.base.load_param(model_path)

            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # disentangle branch
        self.proj_layer = nn.Sequential(nn.Conv2d(self.in_planes, self.in_planes, kernel_size=5, stride=2), nn.ReLU(), nn.Conv2d(self.in_planes,self.in_planes, kernel_size=3, stride=2), nn.Tanh())
        self.proj_layer.apply(weights_init_kaiming)

        self.feature_layer = nn.Sequential(nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), nn.ReLU(), nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1))
        self.feature_layer.apply(weights_init_kaiming)

        self.bg_layer = nn.Sequential(nn.Conv2d(self.in_planes, self.in_planes//16, kernel_size=3, stride=1, padding=1, padding_mode="reflect"), nn.ReLU(), nn.Conv2d(self.in_planes//16, bg_dim, kernel_size=3, stride=1, padding=1))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


    def forward(self, x, label=None, dis=False):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        disentangle = self.feature_layer(x)
        proj_feature_map = self.proj_layer(x)
        proj_feature_map = nn.functional.avg_pool2d(proj_feature_map, proj_feature_map.shape[2:4])
        proj_feature_map = proj_feature_map.view(proj_feature_map.shape[0], -1)

        bg_feature_map = self.bg_layer(x)

        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if not dis:
                if self.cos_layer:
                    cls_score = self.arcface(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat  # global feature for triplet loss
            else:
                if self.cos_layer:
                    cls_score = self.arcface(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat, disentangle, proj_feature_map, bg_feature_map   # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    cutter = CutterNet(cfg)
    taker = TakerNet(cfg, num_class)
    imgdecoder = ImgDecoder(cfg)
    return model, imgdecoder, cutter, taker
