from __future__ import print_function, absolute_import, division
import time
import numpy as np
import random
import torch
import torch.nn as nn
import os
from .utils.meters import AverageMeter
from .FFT import amplitude_spectrum_mix

class Trainer(object):
    def __init__(self, args, model, memory, criterion, num_classes):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.num_classes = num_classes
        self.args = args

    def apply_topk_disturbance(self, img, weight, K = 10, probability = 0.2):

        TopK_ALL = torch.argsort(weight, dim=1, descending=True)

        TopK = TopK_ALL[:, :K]
        batch_index = 0
        img_original = img.permute(0, 2, 3, 1)

        for index in TopK:
            if random.random() <= probability:
                for j in range(TopK.shape[1]):
                    patch_index_h = int(index[j] // 16)
                    patch_index_w = int(index[j] % 16)
                    img_src = img_original[batch_index, 16 * patch_index_h:16 * (1 + patch_index_h),
                              8 * patch_index_w:8 * (1 + patch_index_w), :]

                    random_index = torch.randint(0, img_original.size(0), (1,)).item()
                    random_h = torch.randint(0, 16, (1,)).item()
                    random_w = torch.randint(0, 16, (1,)).item()
                    img_random = img_original[random_index, 16 * random_h:16 * (1 + random_h),
                                 8 * random_w:8 * (1 + random_w), :]
                    img_src_random = amplitude_spectrum_mix(img_src, img_random, alpha=1)
                    img_original[batch_index, 16 * patch_index_h:16 * (1 + patch_index_h),
                    8 * patch_index_w:8 * (1 + patch_index_w), :] = img_src_random
            batch_index += 1
        return img_original.permute(0, 3, 1, 2).contiguous()

    def train(self, epoch, data_loaders, data_loaders_Imagenet, optimizer, print_freq=10, train_iters=400):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        source_count = len(data_loaders)
        print('source_count={}'.format(source_count))
        end = time.time()
        for i in range(train_iters):

            if True:
                data_loader_index = [i for i in range(source_count)] #  [0, 1, 2]
                batch_data = [data_loaders[i].next() for i in range(source_count)]
                data_time.update(time.time() - end)
                loss_train = 0.
                for t in data_loader_index: # 0 1 2
                    data_time.update(time.time() - end)
                    traininputs = batch_data[t]
                    inputs, targets = self._parse_data(traininputs)

                    # ATP #
                    with torch.no_grad():
                        self.model.eval()
                        _, topk_weight = self.model(inputs)
                    self.model.train()

                    perturbed_inputs = self.apply_topk_disturbance(inputs, weight = topk_weight, K=10, probability = 0.2)  # 扰动操作
                    f_out, tri_features = self.model(perturbed_inputs)
                    loss_s = self.memory[t](f_out, targets).mean()

                    loss_train = loss_train + loss_s


                loss_train = loss_train / source_count

                optimizer.zero_grad()

                loss_train.backward()
                optimizer.step()

                losses.update(loss_train.item())

                with torch.no_grad():
                    for m_ind in range(source_count):
                        imgs, pids = self._parse_data(batch_data[m_ind])
                        f_new, _ = self.model(imgs)
                        self.memory[m_ind].module.MomentumUpdate(f_new, pids)

            # print log#
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'loss_s {:.3f}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              loss_s,
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        imgs = imgs.cuda()
        pids = pids.cuda()

        return imgs, pids

