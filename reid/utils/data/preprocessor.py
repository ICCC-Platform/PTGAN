from __future__ import absolute_import
import random
import glob
import numpy as np
import torch
from scipy import ndimage
from PIL import Image
from reid.utils.data import transforms


class Preprocessor(object):
    def __init__(self, dataset, root=None, with_pose=False, pose_root=None, pid_imgs=None, height=256, width=256,
                 pose_aug='no', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.with_pose = with_pose
        self.pose_root = pose_root
        self.pid_imgs = pid_imgs
        self.height = height
        self.width = width
        self.pose_aug = pose_aug

        normalizer = transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])

        if transform==None:
            self.transform = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.RandomSizedEarser(),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])
        else:
            self.transform = transform
        self.transform_p = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            if not self.with_pose:
                return [self._get_single_item(index) for index in indices]
            else:
                return [self._get_single_item_with_pose(index) for index in indices]
        if not self.with_pose:
            return self._get_single_item(indices)
        else:
            return self._get_single_item_with_pose(indices)

    def _get_single_item(self, index):
        fname, pid, pose, camid, color, type = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid, color, type

    def _get_single_item_with_pose(self, index):
        fname, pid, origin_pose, camid, color, type = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        gtpath = fpath[:-26]
        gtpath = glob.glob(gtpath + '*')
        target_pose = gtpath[random.randint(0, len(gtpath)-1)]
        gtpath = glob.glob(target_pose + '/*.jpg')
        gtpath = gtpath[random.randint(0, len(gtpath)-1)]
        gt_img = Image.open(gtpath).convert('RGB')

        target_pose = gtpath[:-4] + '.txt'
        landmark = self._load_landmark(target_pose, self.height/gt_img.size[0], self.width/gt_img.size[1])
        maps = self._generate_pose_map(landmark)
        gt_img = self.transform_p(gt_img)
        target_pose = int(target_pose[-26])

        return {'origin': img,
                'target': gt_img,
                'posemap': torch.Tensor(maps),
                'target_pose': target_pose,
                'pid': torch.LongTensor([pid]),
                'color': color,
                'type': type,
                'camid': camid,
                'origin_pose': origin_pose,
                }

    def _load_landmark(self, path, scale_h, scale_w):
        landmark = []
        with open(path, 'r') as f:
            landmark_file = f.readlines()
        for i, line in enumerate(landmark_file):
            if i % 2 == 0:
                h0 = int(float(line) * scale_h)
                if h0 < 0:
                    h0 = -1
            else:
                w0 = int(float(line) * scale_w)
                if w0 < 0:
                    w0 = -1
                landmark.append(torch.Tensor([[w0, h0]]))
        landmark = torch.cat(landmark).long()
        # avoid to over fit
        ram = random.randint(0, 19)
        landmark[ram][0] = random.randint(0, self.height-1)
        landmark[ram][1] = random.randint(0, self.width-1)
        return landmark

    def _generate_pose_map(self, landmark, gauss_sigma=5):
        maps = []
        randnum = landmark.size(0)+1
        if self.pose_aug == 'erase':
            randnum = random.randrange(landmark.size(0))
        elif self.pose_aug == 'gauss':
            gauss_sigma = random.randint(gauss_sigma-1, gauss_sigma+1)
        elif self.pose_aug != 'no':
            assert ('Unknown landmark augmentation method, choose from [no|erase|gauss]')
        for i in range(landmark.size(0)):
            map = np.zeros([self.height, self.width])
            if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
                map[landmark[i, 0], landmark[i, 1]] = 1
                map = ndimage.filters.gaussian_filter(map, sigma=gauss_sigma)
                map = map / map.max()
            maps.append(map)
        maps = np.stack(maps, axis=0)
        return maps
