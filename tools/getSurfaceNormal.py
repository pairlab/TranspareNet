import numpy as np
import cv2
import os
import OpenEXR
import Imath
import math
import torch
import torch.nn as nn
from saic_depth_completion.data.datasets.franka_scan_separate_by_num import FrankaScan

def depth_to_normal(d_im, save = False, save_path = '',factor = 320, length_ratio = 320):
    zy, zx = np.gradient(d_im)  

    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    # zx = cv2.Sobel(length_ratio*d_im, cv2.CV_64F, 1, 0, ksize=5)     
    # zy = cv2.Sobel(length_ratio*d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-factor*zx, -factor*zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    # normal += 1
    # normal /= 2

    # Gaussian blur
    kX = 15
    kY = 15
    normal = cv2.GaussianBlur(normal, (kX, kY), 0)
    # cv2.imshow("Gaussian ({}, {})".format(kX, kY), normal)
    # cv2.waitKey(0)

    
    if save:
        normal_save = normal+1
        normal_save /= 2
        normal_save = normal_save*255
        # print(normal_save.shape)
        normal_save[d_im>1,:] = np.array([128,0,128])
        cv2.imwrite(save_path, normal_save[:, :, ::-1])
    
    
    
    return normal

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector,axis = 2)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def degrees_difference(normal_1, normal_2):
    radians_difference = angle_between(normal_1, normal_2)
    degrees_difference = np.degrees(radians_difference)
    return degrees_difference

def metric_calculator(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth
    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (3, height, width).
        mask (tensor): Optional mask of area where loss is to be calculated. All other pixels are ignored.
                       Shape: (height, width), dtype=uint8
    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees
    """
    if len(input_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=0))
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    # print(total_valid_pixels)
    # TODO: How to deal with a case with zero valid pixels?
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    loss_cos = cos(input_vec, target_vec)
    # print(loss_cos)
    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels

def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr

if __name__ =='__main__':
    save = False
    pred_normal_compute = True
    dataset_root = ''
    test_datasets = {
        "test_frankascanv2_1": FrankaScan(split='test', processed = False, model_segment=False,num_vessels = 1,  pcc_trans10k = False),
        "test_frankascanv2_2": FrankaScan(split='test', processed = False, model_segment=False,num_vessels = 2,  pcc_trans10k = False),
        "test_frankascanv2_3": FrankaScan(split='test', processed = False, model_segment=False,num_vessels = 3,  pcc_trans10k = False),
        "test_frankascanv2_4": FrankaScan(split='test_val', processed = False, model_segment=False,num_vessels = 3,  pcc_trans10k = False)}

    # Do computation


    for dataset in test_datasets.keys():
        running_mean = []
        running_median = []
        running_percentage1 = []
        running_percentage2 = []
        running_percentage3 = []
        for data_index in range(len(test_datasets[dataset])):
            data = test_datasets[dataset][data_index]
            data_path, _ = os.path.split(test_datasets[dataset].color_name[data_index])
            # print('processing: ', data_path)
            pred_depth_path = os.path.join(data_path,'transpareNet_depth.exr')

            pred_depth_data = exr_loader(pred_depth_path, ndim=1)
            pred_depth_data[np.isnan(pred_depth_data)] = 0.0
            pred_depth_data[np.isinf(pred_depth_data)] = 0.0

            raw_depth_data = data['raw_depth']
            gt_depth_data = data['gt_depth']
            mask_data = data['gt_mask']
            mask_data = torch.reshape(mask_data,(480,640))
            # print(pred_depth_data.shape)
            pred_normal = depth_to_normal(pred_depth_data, save = save, save_path = os.path.join(data_path, 'pred_normal_gb15.png'))
            # print(gt_depth_data.numpy().shape)
            gt_normal = depth_to_normal(gt_depth_data.numpy().reshape((480,640)),save = save, save_path = os.path.join(data_path, 'gt_normal_gb15.png'))
            raw_normal = depth_to_normal(raw_depth_data.numpy().reshape((480,640)),save = save, save_path = os.path.join(data_path, 'raw_normal_gb15.png'))
            # print(mask_data.shape)

            # Convert raw_normal and gt_normal to tensors
            pred_normal = torch.tensor(pred_normal.transpose([2, 0, 1]),dtype=torch.float32)
            # print(pred_normal)
            gt_normal = torch.tensor(gt_normal.transpose([2, 0, 1]),dtype=torch.float32)

            # print(gt_normal.shape)


            if pred_normal_compute:

                loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = metric_calculator(
                    pred_normal, gt_normal, mask=~mask_data.bool()) #.repeat(3,1,1))
            else:
                raw_normal = torch.tensor(raw_normal.transpose([2, 0, 1]),dtype=torch.float32)
                loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels = metric_calculator(
                    raw_normal, gt_normal, mask=~mask_data.bool()) #.repeat(3,1,1))
            running_mean.append(loss_deg_mean.item())
            running_median.append(loss_deg_median.item())
            running_percentage1.append(percentage_1.item())
            running_percentage2.append(percentage_2.item())
            running_percentage3.append(percentage_3.item())
            
            
        print('======================')
        print('mean: ', np.average(np.array(running_mean)))
        print('median: ', np.average(np.array(running_median)))
        print('percentage1: ', np.average(np.array(running_percentage1)))
        print('percentage2: ', np.average(np.array(running_percentage2)))
        print('percentage3: ', np.average(np.array(running_percentage3)))


        

    # Save Depth
    # Compute score

