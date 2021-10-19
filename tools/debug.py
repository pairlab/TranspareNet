import torch

import grnet_point_cloud_completion.utils.data_loaders
from configs.grnet.config import cfg


train_dataset_loader = grnet_point_cloud_completion.utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        grnet_point_cloud_completion.utils.data_loaders.DatasetSubset.TRAIN),
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=grnet_point_cloud_completion.utils.data_loaders.collate_fn,
                                                    pin_memory=False,
                                                    shuffle=False,
                                                    drop_last=True)
for t, i, p in train_data_loader:
    part = p['partial_cloud']
    gt = p['gt_cloud']
    torch.sum(part, dim=3)
