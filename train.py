import os, sys
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from reid import datasets
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomPairSampler
from reid.utils.data import transforms as T
from reid.evaluators import CascadeEvaluator

from gan.options import Options
from gan.utils.visualizer import Visualizer
from gan.model import Model
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_data(name, data_dir, height, width, batch_size, workers, pose_aug):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)

    # use combined trainval set for training as default
    train_loader = DataLoader(
        Preprocessor(dataset.train, root=dataset, with_pose=True, height=height, width=width, pose_aug=pose_aug),
        sampler=RandomPairSampler(dataset.train, neg_pos_ratio=3),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    return dataset, train_loader


def main():
    opt = Options().parse()
    dataset, train_loader = get_data(opt.dataset, opt.dataroot, opt.height, opt.width, opt.batch_size, opt.workers,
                                     opt.pose_aug)

    dataset_size = len(dataset.train)*4
    print('#training images = %d' % dataset_size)

    model = Model(opt)
    visualizer = Visualizer(opt)

    total_steps = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.reset_model_status()

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_step == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    main()
