# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import torch
import shutil
import grnet_point_cloud_completion.utils.data_loaders

from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from grnet_point_cloud_completion.core.test import test_net
from grnet_point_cloud_completion.extensions.chamfer_dist import ChamferDistance
from grnet_point_cloud_completion.extensions.gridding_loss import GriddingLoss
from grnet_point_cloud_completion.models.grnet import GRNet
from grnet_point_cloud_completion.utils.average_meter import AverageMeter

import json
import os


def get_list(path):
    pt_list = []
    for _, dirs, _ in os.walk(path):
        for dirname in sorted(dirs):
            for _, _, files in os.walk(os.path.join(path, dirname)):
                for filename in files:
                    if 'depth2pcd_GT_' in filename:
                        name, extension = os.path.splitext(filename)
                        obj_idx = name[-1]  # the last char is the object index
                        pt_list.append("%s-%s" % (dirname, obj_idx))
    return pt_list


def createjson():
    train_list = get_list("./datasets/frankascanv2/train")
    test_list = get_list("./datasets/frankascanv2/test")

    frankascan = [{
        "taxonomy_id": "frankascanv2",
        "taxonomy_name": "FrankaScan-Beaker",
        "test": test_list,
        "train": train_list,
        "val": []
    }]
    with open('./datasets/FrankaScanv2.json', 'w') as outfile:
        json.dump(frankascan, outfile)


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = grnet_point_cloud_completion.utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = grnet_point_cloud_completion.utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    print('train_dataset_loader')
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        grnet_point_cloud_completion.utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=grnet_point_cloud_completion.utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    print('test_dataset_loader')
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        grnet_point_cloud_completion.utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  collate_fn=grnet_point_cloud_completion.utils.data_loaders.collate_fn,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    grnet = GRNet(cfg)
    grnet.apply(grnet_point_cloud_completion.utils.helpers.init_weights)
    logging.debug('Parameters in GRNet: %d.' % grnet_point_cloud_completion.utils.helpers.count_parameters(grnet))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        # best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        # best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        grnet.load_state_dict(checkpoint['grnet'])
        # logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        grnet.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)

        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            try:
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = grnet_point_cloud_completion.utils.helpers.var_or_cuda(v)
                sparse_ptcloud, dense_ptcloud = grnet(data)
            except Exception as e:

                i = int(str(e))
                print(model_ids[i])
                shutil.move(f'datasets/frankascanv2/train/{model_ids[i].split("-")[0]}', 'datasets/frankascanv2/error')
                createjson()
                exit(0)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            _loss = sparse_loss + dense_loss
            losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (
                             epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(),
                             data_time.val(),
                             ['%.4f' % l for l in losses.val()]))


        grnet_lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, grnet)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()
