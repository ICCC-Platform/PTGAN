import pickle
import os
import os.path as osp
import time
import warnings

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from tqdm import tqdm
# from skimage.measure import compare_ssim
from skimage import measure
from skimage.io import imsave
from torch.utils.data import DataLoader

# from gan.utils.visualizer import Visualizer
from gan.model import Model
from gan.options import Options
# from gan.utils.fid_score import FID_score
# from gan.utils.inception_score import get_inception_score
from testset import ImageDataset, _pluck
from torchvision import transforms

warnings.simplefilter("ignore", UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# wandb.login(key="d30747b0b1afc6aaca38c23a7d702a8ef2c533f1")


def get_data(data_dir):
    gallery_root = osp.join(data_dir, 'test')
    gallery_data, gallery_path_dict = _pluck(gallery_root, False)
    # gallery_pose_list = get_pose_list(gallery_root)
    gallery_dataset = ImageDataset(gallery_data, gallery_path_dict, height=224, width=224, pose_aug='gauss')

    # use combined trainval set for training as default
    gallery_loader = DataLoader(gallery_dataset, batch_size=1, pin_memory=True)

    return gallery_loader, gallery_data  # easer query


def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image / 255.0 - 0.5) - 2 * (generated_image / 255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    # n*H*W*C
    ssim_subtotal = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = measure.compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, multichannel=True,
                                    data_range=generated_image.max() - generated_image.min())
        ssim_subtotal.append(ssim)
    # print(np.mean(ssim_subtotal))
    return np.mean(ssim_subtotal)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))




# def log_image_table(image, target_poses, fake_images, target_image,fullid):
#     wandb.init(
#         # Set the project where this run will be logged
#         project="PTGAN_testing")
#     num = len(target_poses)
#     img_list = []
#     img_list.append(image.to("cpu").permute(1, 2, 0))
#     for i in range(target_image.shape[0]):
#         img = target_image[i].permute(1, 2, 0)
#         img_list.append(fake_images[i].to("cpu").permute(1, 2, 0))

#         img_list.append(target_image[i].to("cpu").permute(1, 2, 0))

#     name = [fullid]
#     for i in range(num):
#         name += [f"fake_pose_{target_poses[i]}"]
#         name += [f"target_pose_{target_poses[i]}"]
#     for i, img in enumerate(img_list):
#         nor_img = wandb.Image((img.numpy() * 0.5 + 0.5) * 255, caption="name:{}".format(name[i]))
#         wandb.log({fullid[0]: nor_img})

def save_pil(save_dir, fake_images, target_images):
    global file_dict
    for fake, target in zip(fake_images, target_images):
        fake = transforms.ToPILImage()(fake).convert('RGB')
        save_name = f"/mnt/Nami/PTGAN/{save_dir}/{len(file_dict)}.jpg"
        file_dict[save_name] = target
        fake.save(save_name)


def main():

    opt = Options().parse()
    print(opt.save_dir)
    print(f"/mnt/Nami/PTGAN/{opt.save_dir}")
    print(f'/mnt/Nami/PTGAN/{opt.save_dir}_label')
    gallery_loader, gallery_data = get_data('/home/ANYCOLOR2434/AICITY2021_Track2_DMT/AIC21/veri_pose')
    device = "cuda"
    # dataset_size = len(dataset.train)*4
    print('#testing images = %d' % len(gallery_data))
    if device and torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    model = Model(opt)

    # visualizer = Visualizer(opt)

    total_steps = 0
    ssim_score_list = []
    global file_dict
    file_dict = {}
    with torch.no_grad():
        for epoch in range(1):
            epoch_start_time = time.time()
            epoch_iter = 0
            model.reset_model_status()
            # wandb.init(
            #     # Set the project where this run will be logged
            #     project="PTGAN_testing",
            #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            #     name=f"experiment_{opt.dataset}_{epoch}",
            #     # Track hyperparameters and run metadata
            #     config={
            #         "learning_rate": 0.02,
            #         "architecture": "opt",
            #         "dataset": opt.dataset,
            #         "netE-pretrain": opt.netE_pretrain,
            #         "netG-pretrain": opt.netG_pretrain,
            #         'netDp_pretrain': opt.netDp_pretrain,
            #         'netDi_pretrain': opt.netDi_pretrain,
            #         'height': opt.height,
            #         'width': opt.width
            #     })
            for i, data in enumerate(tqdm(gallery_loader)):
                # iter_start_time = time.time()
                # visualizer.reset()
                gallery_img = data['origin'].to(device)

                target_poses = data['posemap_tensor'].to(device)
                target_pose_list = data['target_pose_list']
                if len(target_pose_list) == 0:
                    continue
                # ori_img = gallery_img.squeeze(0)
                target_poses = target_poses.squeeze(0)
                fake_imgs = model.generate(gallery_img, target_poses)
                # img = fake_imgs[0].cpu().numpy() * 0.5 + 0.5
                # img = img.transpose((1, 2, 0)) * 255
                # img2 = Image.fromarray(img.astype('uint8')).convert('RGB')
                # img2.show()
                # target_imgs = data['gt_img_tensor'].squeeze(0)
                # log_image_table(ori_img, target_poses, fake_imgs, target_imgs, data['fullid'])
                # model.set_input(data)
                fake_imgs_np = fake_imgs.cpu() * 0.5 + 0.5                

                # fake_imgs_np = fake_imgs_np.transpose((0, 2, 3, 1)) * 255
                # fake_imgs_np = fake_imgs_np.astype('uint8')
                # target_imgs_np = target_imgs.cpu() * 0.5 + 0.5
                # target_imgs_np = transforms.ToPILImage()(target_imgs_np).convert('RGB')
                # target_imgs_np = target_imgs_np.transpose((0, 2, 3, 1)) * 255
                # target_imgs_np = target_imgs_np.astype('uint8')

                save_pil(opt.save_dir, fake_imgs_np, target_pose_list)

                # ssim = ssim_score(fake_imgs_np, target_imgs_np)
                # ssim_score_list.append(ssim)
                # fid = FID_score()
                # fid_score = fid.calculate_fid_images(fake_imgs_np, target_imgs_np)
                # fid_score_list.append(fid_score)
            
            with open(f'/mnt/Nami/PTGAN/{opt.save_dir}_label', 'wb') as f:
                pickle.dump(file_dict, f)
            
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    # ssim_1score = np.mean(ssim_score_list)
    # fid_1score = np.mean(fid_score_list)
    # print("ssim_score",ssim_1score)
    # print("fid_score:", fid_1score)
    # Mark the run as finished
    # wandb.finish()


if __name__ == '__main__':
    main()
