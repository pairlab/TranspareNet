import os
import numpy as np

from grnet_point_cloud_completion.utils import IO


def rms(path, gt_name, pred_name):
    """ Compute the average root-mean-squared error of predicted depth images.
    """
    errors = []
    for subdir, dirs, files in os.walk(path):
        for dirname in sorted(dirs):
            depth_gt = IO.get(os.path.join(path, '%s/%s' % (dirname, gt_name)))
            depth_pred = IO.get(os.path.join(path, '%s/%s' % (dirname, pred_name)))
            diff = np.where(depth_pred > 0, depth_pred - depth_gt, 0)
            errors.append(np.sqrt(np.mean(diff**2)))
    return np.array(errors).mean(), np.array(errors).sum()


if __name__ == '__main__':
    # path = './frankascan-cleargrasp-norm/test/'
    # path = './frankascan-norm/test/'
    # path = './frankascan-recenter/test/'
    path = './frankascan-gtpcd/test/'
    gt_name = 'detph_GroundTruth.exr'
    pred_name = 'depth_pred.exr'

    rmse = rms(path, gt_name, pred_name)
    print('Root mean squared error:', rmse)
