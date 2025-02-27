import os.path
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from gan.model import Model
from gan.options import Options
from reid.utils.data import transforms


def _load_landmark(img_path, scale_h, scale_w):
    landmark = []
    with open(img_path, 'r') as f:
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
    landmark[ram][0] = random.randint(0, 224 - 1)
    landmark[ram][1] = random.randint(0, 224 - 1)
    return landmark


def _generate_pose_map(landmark, gauss_sigma=5):
    maps = []
    randnum = landmark.size(0) + 1
    gauss_sigma = random.randint(gauss_sigma - 1, gauss_sigma + 1)
    for i in range(landmark.size(0)):
        map = np.zeros([224, 224])
        if landmark[i, 0] != -1 and landmark[i, 1] != -1 and i != randnum:
            map[landmark[i, 0], landmark[i, 1]] = 1
            map = gaussian_filter(map, sigma=gauss_sigma)
            map = map / map.max()
        maps.append(map)
    maps = np.stack(maps, axis=0)

    return maps


def _pluck(keypoints:list[Path], images:list[Path]):
    normalizer = transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
    transform = transforms.Compose([
        transforms.RectScale(224, 224),
        transforms.ToTensor(),
        normalizer,
    ])

    input_images = torch.FloatTensor([])
    pose_map = torch.FloatTensor([])
    for source_path, keypoint in zip(images, keypoints):
        source = Image.open(source_path).convert('RGB')
        source = transform(source)

        target = Image.open(keypoint.parent/f"{keypoint.stem}.jpg").convert('RGB')
        landmark = _load_landmark(keypoint, 224 / target.size[0], 224 / target.size[1])
        pose_map = torch.cat((pose_map, torch.from_numpy(_generate_pose_map(landmark)).unsqueeze(0)), 0)
        input_images = torch.cat((input_images, source.unsqueeze(0)), 0)
    return input_images, pose_map.type(torch.FloatTensor)



if __name__ == '__main__':

    # inferen_device = torch.device('cpu')
    inferen_device = torch.device('cuda', index=0)

    test_path = Path("demo_images")
    data_dir = [test_path/'0420_c008_00063410_0.jpg',
                test_path/'0273_c003_00036570_0.jpg',
                test_path/'0417_c003_00088750_0.jpg',
                test_path/'0188_c001_00050125_1.jpg']
    keypoints = [test_path/'0420_c009_00063470_0.txt',
                 test_path/'0273_c013_00040995_0.txt',
                 test_path/'0417_c010_00088060_0.txt',
                 test_path/'0188_c017_00047990_0.txt']
    input_images, pose_map = _pluck(keypoints, data_dir)

    opt = Options().parse()
    opt.inference_device = inferen_device
    model = Model(opt)
    print(input_images.size(), pose_map.size())
    fake_imgs = model.generate(input_images.to(opt.inference_device), pose_map.to(opt.inference_device))
    fake_imgs_np = fake_imgs.cpu() * 0.5 + 0.5
    save_dir = 'noise_cycle'
    save_root = Path("experiment")
    save_path = save_root/save_dir
    save_path.mkdir(parents=True, exist_ok=True)

    for i, fake in enumerate(fake_imgs_np):
        print(fake.shape)
        fake:Image.Image = transforms.ToPILImage()(fake).convert('RGB')
        print(save_path/f'40_{i}.jpg')
        fake.save(save_path/f'40_{i}.jpg')
