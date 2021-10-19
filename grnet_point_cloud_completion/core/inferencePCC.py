import torch

import numpy as np
from grnet_point_cloud_completion.datasets.img2pcd import img2pcdHelper
from grnet_point_cloud_completion.datasets.pcd2img import pcd2imgHelper


def inference_pcc(grnet, n_points, K, batch):
    mask = batch["mask"][0].cpu().detach().numpy()[0]
    depth = batch["raw_depth"][0].cpu().detach().numpy()[0]
    # Switch models to evaluation mode
    grnet.eval()
    INV_K = np.linalg.inv(K)
    maxdis, pcds, centers = img2pcdHelper(mask, depth, INV_K)
    complete_pcds = []
    for i in range(len(pcds)):
        ptcloud = pcds[i]
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        ptcloud = torch.from_numpy(ptcloud.copy()).float()
        data = {'partial_cloud': ptcloud}
        try:
            sparse_ptcloud, dense_ptcloud = grnet(data)
            dense_ptcloud = dense_ptcloud.squeeze().cpu().detach().numpy()
        except:
            dense_ptcloud = ptcloud.squeeze().cpu().detach().numpy()

        complete_pcds.append(dense_ptcloud)
    pred_depth = pcd2imgHelper(mask, depth, K, complete_pcds, maxdis, centers)
    batch["raw_depth"][0] = torch.from_numpy(pred_depth.copy()).float().cuda()
    return batch
