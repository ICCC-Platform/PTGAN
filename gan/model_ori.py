import os
import sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.optim as optim
import gan.utils.util as util
from gan.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, remove_module_key, \
    set_bn_fix, get_scheduler, print_network, OrthogonalEncoder, IDDiscriminator
from gan.gan_losses import GANLoss


class Model(object):

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)

        self._init_models()

        print('---------- Networks initialized -------------')
        print_network(self.net_E)
        print_network(self.net_G)
        print_network(self.net_Di)
        print_network(self.net_Dp)
        print('-----------------------------------------------')

    def _init_models(self):
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                         dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode,
                                         connect_layers=self.opt.connect_layers,
                                         remove_noise=self.opt.remove_noise)
        self.net_E = OrthogonalEncoder()
        print(self.net_E)
        self.net_Di = IDDiscriminator(self.opt.id_class)
        self.net_Dp = NLayerDiscriminator(3+20, norm_layer=self.norm_layer)

        self._load_state_dict(self.net_E, self.opt.netE_pretrain)
        self._load_state_dict(self.net_G, self.opt.netG_pretrain)
        self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
        self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)

        self.net_E = nn.DataParallel(self.net_E).to('cuda')
        self.net_G = nn.DataParallel(self.net_G).to('cuda')
        self.net_Di = nn.DataParallel(self.net_Di).to('cuda')
        self.net_Dp = nn.DataParallel(self.net_Dp).to('cuda')

    def reset_model_status(self):
        self.net_G.eval()
        self.net_Dp.eval()
        self.net_E.eval()
        self.net_Di.eval()

    def _load_state_dict(self, net, path):
        state_dict = remove_module_key(torch.load(path, map_location=torch.device('cpu')))
        net.load_state_dict(state_dict)

    # def enocder(self, img):
    #     noise = torch.randn(img.shape[0], self.opt.noise_feature_size)
    #     # noise = noise.to('cuda')
    #     outputs = self.net_E(img)
    #     id_feature = outputs[1].view(outputs[0].size(0), outputs[0].size(1), 1, 1)
    #     return noise, id_feature, torch.argmax(outputs[2], dim=1), torch.argmax(outputs[5], dim=1)

    def enocder(self, img):
        noise = torch.randn(img.shape[0], 128)
        # noise = noise.to('cuda')
        outputs = self.net_E(img)
        id_feature = outputs[1].view(outputs[0].size(0), outputs[0].size(1), 1, 1)
        return noise, id_feature, torch.argmax(outputs[2], dim=1)

    def generate(self, query_img, target_pose, target_pose_list):
        # pose_feature, id_feature, pose, id, color, type
        noise, id_feature, q_pose = self.enocder(query_img)
        # pose_feature, id_feature, pose, id, color, type
        if id_feature.shape[0] != target_pose.shape[0]:
            id_feature = id_feature.repeat(target_pose.shape[0], 1, 1, 1)
            noise = noise.repeat(target_pose.shape[0], 1)
        try:
            if target_pose.shape[0] == 0 or id_feature.shape[0] == 0 or noise.shape[0] == 0:
                print(target_pose_list)
            fake = self.net_G(target_pose, id_feature, noise.view(noise.size(0), noise.size(1), 1, 1))
        except:
            print(target_pose_list)
        return fake

    # def generate(self, target_pose, id_feature, noise):
    #     # pose_feature, id_feature, pose, id, color, type
    #     fake = self.net_G(target_pose, id_feature, noise.view(noise.size(0), noise.size(1), 1, 1))
    #     return fake

    def get_pose(self, img):
        # pose_feature, id_feature, pose, id, color, type
        outputs = self.net_E(img)
        return outputs[2]

    def get_current_visuals(self):
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1] = 1
        map = util.tensor2im(torch.unsqueeze(map, 1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])
