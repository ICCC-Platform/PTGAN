from __future__ import print_function, absolute_import
import time
import numpy as np
import torch
from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
# import visdom
# vis = visdom.Visdom(env='GAN_train-stage-2', port=8097)


class BaseTrainer(object):
    def __init__(self, model, criterion, num_classes=0, num_instances=4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_instances = num_instances

    def train(self, epoch, data_loader, optimizer, base_lr=0.1, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        vis.line(
            Y=np.column_stack((np.array(losses.avg), np.array(losses.avg))),
            X=np.column_stack((epoch, epoch)),
            win='Learning curve',
            update='append',
            opts={
                'title': 'Learning curve',
            }
        )
        vis.line(
            Y=np.column_stack((precisions.avg.cpu().numpy(), precisions.avg.cpu().numpy())),
            X=np.column_stack((epoch, epoch)),
            win='accuracy curve',
            update='append',
            opts={
                'title': 'accuracy curve',
            }
        )

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [imgs1, imgs2]
        targets = (pids1 == pids2).long().cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        _, _, outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data)
        return loss, prec1[0]
