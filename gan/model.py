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
from gan.networks_copy import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, remove_module_key, \
    set_bn_fix, get_scheduler, print_network, OrthogonalEncoder, IDDiscriminator
from gan.gan_losses import GANLoss


class Model(object):

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)
        print('Networks initializing',end="...")
        self._init_models()

        if self.opt.stage < 3:
            self._init_losses()
            self._init_optimizers()

        
        #print_network(self.net_E)
        #print_network(self.net_G)
        #print_network(self.net_Di)
        #print_network(self.net_Dp)
        print('done')

    def _init_models(self):
        
        assert self.opt.stage in {1,2,3}, "unknown stage"
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                         dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode,
                                         remove_noise=self.opt.remove_noise)
        self.net_E = OrthogonalEncoder()
        
        if self.opt.stage < 3:
            self.net_Di = IDDiscriminator(self.opt.id_class)
            self.net_Dp = NLayerDiscriminator(3+20, norm_layer=self.norm_layer)
            
            
        if self.opt.stage == 1:
            init_weights(self.net_G)
            init_weights(self.net_Dp)
            state_dict = remove_module_key(torch.load(self.opt.netE_pretrain))
            model_dict = self.net_E.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            self.net_E.load_state_dict(model_dict)
            state_dict = remove_module_key(torch.load(self.opt.netE_pretrain))
            model_dict = self.net_Di.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            #self.net_Di.load_state_dict(model_dict)

        else:
            if self.opt.stage == 3:
                print(f"inference stage, using :{self.opt.netE_pretrain} and {self.opt.netG_pretrain}")
            
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            
            if self.opt.stage == 2:
                self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
                self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
       
        if self.opt.stage < 3:
            self.net_E = nn.DataParallel(self.net_E).to('cuda')
            self.net_G = nn.DataParallel(self.net_G).to('cuda')
            self.net_Di = nn.DataParallel(self.net_Di).to('cuda')
            self.net_Dp = nn.DataParallel(self.net_Dp).to('cuda')
        else:
            self.net_E = self.net_E.to(self.opt.inference_device)
            self.net_G = self.net_G.to(self.opt.inference_device)
            self.reset_model_status()

    def reset_model_status(self):
        self.net_E.eval()
        self.net_G.eval()
        if self.opt.stage < 3:
            self.net_Di.eval()
            self.net_Dp.eval()
            # self.net_E.apply(set_bn_fix)
            # self.net_Di.apply(set_bn_fix)


    def _load_state_dict(self, net, path):
        state_dict = remove_module_key(torch.load(path, map_location='cpu'))
        net.load_state_dict(state_dict)

    def enocder(self, img):
        noise = torch.randn(img.shape[0], 128)
        # noise = noise.to('cuda')
        outputs = self.net_E(img)
        noise = noise.to(outputs[1].device)
        id_feature = outputs[1].view(outputs[0].size(0), outputs[0].size(1), 1, 1)
        return noise, id_feature, torch.argmax(outputs[2], dim=1)

    def generate(self, query_img, target_pose):
        # pose_feature, id_feature, pose, id, color, type
        noise, id_feature, q_pose = self.enocder(query_img)
        # pose_feature, id_feature, pose, id, color, type
        if id_feature.shape[0] != target_pose.shape[0]:
            id_feature = id_feature.repeat(target_pose.shape[0], 1, 1, 1)
            noise = noise.repeat(target_pose.shape[0], 1)
        #print(target_pose.device, id_feature.device, noise.device)
        fake = self.net_G(target_pose, id_feature, noise.view(noise.size(0), noise.size(1), 1, 1))
        return fake

    def _init_losses(self):
        if self.opt.smooth_label:
            self.criterionGAN_D = GANLoss(smooth=True).to('cuda')
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.criterionGAN_D = GANLoss(smooth=False).to('cuda')
            self.rand_list = [False]
        self.criterionGAN_G = GANLoss(smooth=False).to('cuda')
        self.criterion = nn.CrossEntropyLoss().to('cuda')

    def _init_optimizers(self):
        if self.opt.stage == 1:
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = optim.SGD(self.net_Di.parameters(), lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = optim.SGD(self.net_Dp.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage == 2:
            param_groups = [{'params': self.net_E.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_G.parameters(), 'lr_mult': 0.1}]
            self.optimizer_G = optim.Adam(param_groups, lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = optim.SGD(self.net_Di.parameters(), lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = optim.SGD(self.net_Dp.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage == 3:
            param_groups = [{'params': self.net_E.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_G.parameters(), 'lr_mult': 0.1}]
            self.optimizer_G = optim.Adam(param_groups, lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = optim.SGD(self.net_Di.parameters(), lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = optim.SGD(self.net_Dp.parameters(), lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)

        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))

    def set_input(self, input):
        input1, input2 = input
        labels = (input1['pid'] == input2['pid']).long()
        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)

        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        pid = torch.cat([input1['pid'], input2['pid']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        target_pose = torch.cat([input1['target_pose'], input2['target_pose']])
        origin_pose = torch.cat([input1['origin_pose'], input2['origin_pose']])
        color = torch.cat([input1['color'], input2['color']])
        type = torch.cat([input1['type'], input2['type']])
        noise = torch.cat((noise, noise))

        self.origin = origin.to('cuda')
        self.target = target.to('cuda')
        self.pid = pid.squeeze().to('cuda')
        self.posemap = posemap.to('cuda')
        self.target_pose = target_pose.squeeze().to('cuda')
        self.origin_pose = origin_pose.squeeze().to('cuda')
        self.color = color.squeeze().to('cuda')
        self.type = type.squeeze().to('cuda')
        self.labels = torch.full((origin.size(0),), self.opt.id_class).long().to('cuda')
        self.orthogonal_label = torch.full((origin.shape[0],), 0.0).to('cuda')
        self.noise = noise.to('cuda')

    def forward(self):
        A = self.origin
        B_map = self.posemap
        z = self.noise
        # pose_feature, id_feature, pose, id, color, type
        self.outputs = self.net_E(A)
        self.fake = self.net_G(B_map, self.outputs[1].view(self.outputs[0].size(0), self.outputs[0].size(1), 1, 1),
                               z.view(z.size(0), z.size(1), 1, 1))

    def backward_Dp(self):
        real_pose = torch.cat((self.posemap, self.target), dim=1)
        fake_pose = torch.cat((self.posemap, self.fake.detach()), dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Dp = loss_D.item()

    def backward_Di(self):
        real_pose = torch.cat((self.posemap, self.target), dim=1)
        fake_pose = torch.cat((self.posemap, self.fake.detach()), dim=1)
        pred_real = self.net_Di(real_pose)
        pred_fake = self.net_Di(fake_pose)
        if random.choice(self.rand_list):
            loss_D_real = self.criterion(pred_fake, self.pid)
            loss_D_fake = self.criterion(pred_real, self.labels)
        else:
            loss_D_real = self.criterion(pred_real, self.pid)
            loss_D_fake = self.criterion(pred_fake, self.labels)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.loss_Di = loss_D.item()

    def backward_G(self):
        loss_pose = self.criterion(self.outputs[2], self.origin_pose)
        loss_id = self.criterion(self.outputs[3], self.pid)
        loss_color = self.criterion(self.outputs[4], self.color)
        loss_type = self.criterion(self.outputs[5], self.type)
        loss_orthogonal = F.l1_loss(self.outputs[6], self.orthogonal_label)
        loss_encoder = loss_pose + loss_id + loss_color + loss_type + loss_orthogonal
        loss_r = F.l1_loss(self.fake, self.target)

        pred_fake_Di = self.net_Di(torch.cat((self.posemap, self.fake), dim=1))
        pred_fake_Dp = self.net_Dp(torch.cat((self.posemap, self.fake), dim=1))
        loss_G_GAN_Di = self.criterion(pred_fake_Di, self.pid)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)

        loss_G = loss_G_GAN_Di + loss_G_GAN_Dp + loss_r * self.opt.lambda_recon + loss_encoder *\
                 self.opt.lambda_orthogonal
        loss_G.backward()

        self.loss_G = loss_G.item()
        self.loss_encoder = loss_encoder.item()
        self.loss_r = loss_r.item()
        self.loss_G_GAN_Di = loss_G_GAN_Di.item()
        self.loss_G_GAN_Dp = loss_G_GAN_Dp.item()
        self.fake = self.fake

    def optimize_parameters(self):
        self.forward()

        self.optimizer_Di.zero_grad()
        self.backward_Di()
        self.optimizer_Di.step()

        self.optimizer_Dp.zero_grad()
        self.backward_Dp()
        self.optimizer_Dp.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_e', self.loss_encoder),
                            ('G_r', self.loss_r),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])

    def get_current_visuals(self):
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1] = 1
        map = util.tensor2im(torch.unsqueeze(map, 1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])

    def save(self, epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']

