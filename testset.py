import glob

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from reid.utils.data import transforms
from PIL import Image
import random
from gan.options import Options


def _pluck(root, query):
    ret = []
    path_dict = {}

    # pose_list = [[[] for i in range(8)] for j in range(9)]
    if query:
        path = glob.glob(root + '/*.jpg')
        index = -1
        for fname in path:
            pid = int(fname[-24:-20])
            # if index != pid:
            #     index += 1
            camid = int(fname[-18:-15])
            ret.append((fname, pid, camid, -1))
    else:
        path = glob.glob(root + '/*/*/*.jpg')
        frame2trackID = dict()
        with open(root + '/../test_track.txt') as f:
            for track_id, line in enumerate(f.readlines()):
                curLine = line.strip().split(" ")
                for frame in curLine:
                    frame2trackID[frame] = track_id

        for fname in path:
            pid = int(fname[-24:-20])
            camid = int(fname[-18:-15])
            poseid = int(fname[-26:-25])
            fullid = fname[-33:-27]
            # tip:1
            if not path_dict.get(fullid):
                path_dict[fullid] = {}
            path_dict[fullid].setdefault(poseid, []).append(fname)
            # tip:2
            # if not path_dict.get(fullid):
            #     path_dict[fullid] = [[] for _ in range(8)]
            # path_dict[fullid][poseid].append(fname)

            ret.append((fname, pid, camid, frame2trackID[fname[-24:]], poseid, fullid))
    # path_dict = { fullid : {poseid : [fname1, fname2,...] }}
    return ret, path_dict
# /home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose/test/008_4_0/3/0038_c008_00008270_0.txt


def get_pose_list(root):
    pose_list = [[[] for i in range(8)] for j in range(9)]
    path = glob.glob(root + '/*/*/*.jpg')

    for fname in path:
        type = int(fname[-28])
        pose = int(fname[-26])
        pose_file = fname[:-4] + '.txt'
        pose_list[type][pose].append(pose_file)
    return pose_list


class ImageDataset(Dataset):
    def __init__(self, dataset, gallery_path_dict, transform=None, height=224, width=224, pose_aug='gauss'):
        self.height = height
        self.width = width
        self.dataset = dataset
        self.gallery_path_dict = gallery_path_dict
        self.transform = transform
        self.pose_aug = pose_aug
        opt = Options().parse()
        normalizer = transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
        # normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if transform is None:
            self.transform = transforms.Compose([
                                 transforms.RectScale(height, width),
                                 transforms.ToTensor(),
                                 normalizer,
                             ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_single_item_with_pose(index)

    def _get_single_item(self, index):
        fname, pid, pose, camid, color, type = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid, color, type
    def _load_landmark(self, img_path, scale_h, scale_w):
        landmark = []
        path = img_path[:-4] + '.txt'
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
    def _get_single_item_with_pose(self, index):
        target_pose_list = []
        maps_list = []
        fname, pid, camid, trackid, poseid, fullid = self.dataset[index]
        fpath = fname
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        # gallery_path_dict = { fullid : {poseid : [fname1, fname2,...] }}
        target_pose_dict = self.gallery_path_dict.get(fullid)
        exist_pose = list(target_pose_dict.keys())
        exist_pose.remove(poseid)

        gt_img_tensor = torch.FloatTensor([])
        for target_poseid in exist_pose:
            target_pose_path = random.choice(target_pose_dict.get(target_poseid))
            gt_img = Image.open(target_pose_path).convert('RGB')
            landmark = self._load_landmark(target_pose_path, self.height/gt_img.size[0], self.width/gt_img.size[1])
            maps = self._generate_pose_map(landmark)
            
            # landmark_t = _load_landmark(target_pose_path[:-4] + '.txt')
            # pose_maps = _generate_pose_map_tensor_(Smoothing, landmark_t)

            gt_img = self.transform(gt_img)
            
            maps_list.append(maps)
            target_pose_list.append(target_pose_path)
            gt_img_tensor = torch.cat((gt_img_tensor, gt_img.unsqueeze(0)), 0)
        return {'origin': img,
                'pid': pid,
                'camid': camid,
                'trackid': trackid,
                'file_name': fname,
                'poseid': poseid,
                'fullid': fullid,
                'exist_pose': exist_pose,
                'target_pose_list': target_pose_list,
                'posemap_tensor': torch.Tensor(maps_list),
                'gt_img_tensor': gt_img_tensor
                }
