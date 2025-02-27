from inspect import trace
import os
from skimage import transform
from skimage.io import imread, imsave
from skimage.measure import compare_ssim
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
import re
# from fid_score import FID_score
from gan.utils.inception_score import get_inception_score
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
from PIL import Image
from reid.utils.data import transforms


class testDataset(Dataset):
        def __init__(self, fake_path, target_path):
            self.fake_path = fake_path
            self.target_path = target_path

            normalizer = transforms.Normalize(mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500])
            self.transform = transforms.Compose([
                # transforms.RectScale(224, 224),
                transforms.ToTensor(),
                # normalizer
            ])

        def __getitem__(self, index):
            # fake_img = np.uint8(transform.resize(imread(self.fake_path[index]), (224, 224)) * 255)
            fake_img = Image.open(self.fake_path[index])
            fake_img = self.transform(fake_img)
            # target_img = np.uint8(transform.resize(imread(self.target_path[index][0]), (224, 224)) * 255)
            # target_img = Image.open(self.target_path[index][0])
            # target_img = self.transform(target_img)
            return fake_img

        def __len__(self):
            return len(self.fake_path)


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in tqdm(zip(reference_images, generated_images)):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in tqdm(zip(reference_images, generated_images)):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def load_images(fake_path, target_path):
    print('load fake images')
    fake_imgs = [np.uint8(transform.resize(imread(fake), (224, 224)) * 255) for fake in tqdm(fake_path)]
    print('load target images')
    target_img = [np.uint8(transform.resize(imread(target[0]), (224, 224)) * 255) for target in tqdm(target_path)]
    # target_img = [imread(target) for target in target_path]
    print('done')
    return fake_imgs, target_img


def _test():
    print("Loading images...")

    with open("/mnt/Nami/PTGAN/testing_cycle_40_label", 'rb') as f:
        file_dict = pickle.load(f)
    fake_path = list(file_dict.keys())
    target_path = list(file_dict.values())
    test = testDataset(fake_path)

    print("Compute structured similarity score (SSIM)...")
    print("Compute l1 score...")

    fake_img, target_img = load_images(fake_path, target_path)
    structured_score = ssim_score(fake_img, target_img)
    norm_score = l1_score(fake_img, target_img)

    print("SSIM score {:.3f}".format(structured_score))
    print("L1 score  {:.3f}".format(norm_score))

    testDataloader = DataLoader(dataset=test, batch_size=1)
    print("Compute inception score...")
    inception_score = get_inception_score(testDataloader)
    print("inception score  {:.3f}".format(inception_score[0]))


def _testloader():
    print("Loading images...")

    with open("/mnt/Nami/PTGAN/testing_cycle_40_label", 'rb') as f:
        file_dict = pickle.load(f)
    fake_path = list(file_dict.keys())
    target_path = list(file_dict.values())
    test = testDataset(fake_path, target_path)
    # testDataloader = DataLoader(dataset=test, batch_size=1, num_workers=2)
    #
    # print("Compute structured similarity score (SSIM)...")
    # print("Compute l1 score...")
    #
    # score_list = []
    # ssim_score_list = []
    # for generated_image, reference_image in tqdm(testDataloader):
    #     generated_image = generated_image.numpy()[0]
    #     reference_image = reference_image.numpy()[0]
    #     score = np.abs(2 * (reference_image / 255.0 - 0.5) - 2 * (generated_image / 255.0 - 0.5)).mean()
    #     score_list.append(score)
    #
    #     ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
    #                         use_sample_covariance=False, multichannel=True,
    #                         data_range=generated_image.max() - generated_image.min())
    #     ssim_score_list.append(ssim)
    #
    # # fake_img, target_img = load_images(fake_path, target_path)
    # structured_score = np.mean(ssim_score_list)
    # norm_score = np.mean(score_list)
    #
    # print("SSIM score {:.3f}".format(structured_score))
    # print("L1 score  {:.3f}".format(norm_score))

    testDataloader = DataLoader(dataset=test, batch_size=1)
    print("Compute inception score...")
    inception_score = get_inception_score(testDataloader)
    print("inception score  {:.3f}".format(inception_score[0]))


if __name__ == "__main__":
    _testloader()