import os
import time
import datetime
import torch
import torch.nn.functional as nnf
from tqdm import tqdm

import matplotlib.pyplot as plt

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.utils import visualize

import numpy as np
import OpenEXR
import Imath

def exr_saver(EXR_PATH, ndarr, ndim=3):
    '''Saves a numpy array as an EXR file with HALF precision (float16)
    Args:
        EXR_PATH (str): The path to which file will be saved
        ndarr (ndarray): A numpy array containing img data
        ndim (int): The num of dimensions in the saved exr image, either 3 or 1.
                        If ndim = 3, ndarr should be of shape (height, width) or (3 x height x width),
                        If ndim = 1, ndarr should be of shape (height, width)
    Returns:
        None
    '''
    if ndim == 3:
        # Check params
        if len(ndarr.shape) == 2:
            # If a depth image of shape (height x width) is passed, convert into shape (3 x height x width)
            ndarr = np.stack((ndarr, ndarr, ndarr), axis=0)

        if ndarr.shape[0] != 3 or len(ndarr.shape) != 3:
            raise ValueError(
                'The shape of the tensor should be (3 x height x width) for ndim = 3. Given shape is {}'.format(
                    ndarr.shape))

        # Convert each channel to strings
        Rs = ndarr[0, :, :].astype(np.float16).tostring()
        Gs = ndarr[1, :, :].astype(np.float16).tostring()
        Bs = ndarr[2, :, :].astype(np.float16).tostring()

        # Write the three color channels to the output file
        HEADER = OpenEXR.Header(ndarr.shape[2], ndarr.shape[1])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs, 'G': Gs, 'B': Bs})
        out.close()
    elif ndim == 1:
        # Check params
        if len(ndarr.shape) != 2:
            raise ValueError(('The shape of the tensor should be (height x width) for ndim = 1. ' +
                              'Given shape is {}'.format(ndarr.shape)))

        # Convert each channel to strings
        Rs = ndarr[:, :].astype(np.float16).tostring()

        # Write the color channel to the output file
        HEADER = OpenEXR.Header(ndarr.shape[1], ndarr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "R"])

        out = OpenEXR.OutputFile(EXR_PATH, HEADER)
        out.writePixels({'R': Rs})
        out.close()

def inference(
        model, test_loaders, metrics, save_dir="", logger=None, file_names = None
):

    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in test_loaders.items():
        idx = 0
        logger.info(
            "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
        )
        if file_names is not None:
            split_files = file_names[subset].color_name
        metrics_meter.reset()
        # loop over dataset
        rmse_temp = list()
        for batch in tqdm(loader):
            

            batch = model.preprocess(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                if save_dir:
                    if file_names is not None:
                        B = batch["color"].shape[0]
                        assert B == 1, f'Ensure batchsize is 1, current batchsize is {B}'
                        for it in range(B):
                            file_name = split_files[idx]
                            fig = visualize.figure(
                                batch["color"][it], batch["raw_depth"][it],
                                batch["mask"][it], batch["gt_depth"][it],
                                post_pred[it], close=True
                            )
                            data_folder_path, rgb_name = os.path.split(file_name)
                            exr_saver(EXR_PATH = os.path.join(data_folder_path,'dmlrn_depth.exr'), ndarr = post_pred[it].detach().cpu()[0].numpy(), ndim=1)
                            fig.savefig(
                                os.path.join(data_folder_path,'dmlrn_result_compare.png'), dpi=fig.dpi
                            )

                            idx += 1
                    else:
                        B = batch["color"].shape[0]
                        assert B == 1, f'Ensure batchsize is 1, current batchsize is {B}'
                        for it in range(B):
                            fig = visualize.figure(
                                batch["color"][it], batch["raw_depth"][it],
                                batch["mask"][it], batch["gt_depth"][it],
                                post_pred[it], close=True
                            )
                            fig.savefig(
                                os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
                            )

                            idx += 1
                # Reshape tensors to proper dimension
                # print(batch["mask"])
                # print(post_pred.shape)
                post_pred = nnf.interpolate(post_pred, size=(144,256))
                batch["gt_depth"] = nnf.interpolate(batch["gt_depth"], size=(144,256))

                # NaN goes to zero in gt_depth
                batch['gt_depth'][batch['gt_depth']!= batch['gt_depth']] = 0
                batch['gt_depth'][batch['gt_depth'] == float("Inf")] = 0
                # batch["gt_depth"] = torch.nan_to_num(batch['gt_depth'],nan = 0.0)
                # print(batch['gt_depth'])
                mask_valid_region = (batch['gt_depth'] > 0.01).byte()
                mask_bool = mask_valid_region
                # mask_seg = (nnf.interpolate(batch["mask"], size=(144,256)) <=0.01).byte()
                # # print('mask valid region', torch.sum(mask_valid_region))
                # # print('mask seg',torch.sum(mask_seg))
                # mask_bool = mask_valid_region.__and__(mask_seg)

                # print('max_gt', torch.max(batch["gt_depth"][mask_bool]))
                # print('mean_gt', torch.mean(batch["gt_depth"][mask_bool]))
                # print('max_pred', torch.max(post_pred[mask_bool]))
                # print('mean_pred', torch.mean(post_pred[mask_bool]))
                # print('mask bool',torch.sum(mask_bool))
                # print('mask_valid_region', torch.sum(mask_valid_region))
                # mask_bool = torch.logical_and(mask_valid_region,mask_seg).byte()
                rmse_temp.append(((batch["gt_depth"][mask_bool] - post_pred[mask_bool])**2).mean().sqrt().cpu().detach().numpy())
                try:
                    metrics_meter.update(post_pred[mask_bool], batch["gt_depth"][mask_bool])
                    # metrics_meter.update(post_pred, batch["gt_depth"])
                except:
                    print('ERRRORR IN METER')
                    continue
                # metrics_meter.update(post_pred.masked_fill(mask_bool,0), batch["gt_depth"].masked_fill(mask_bool,0))
                # metrics_meter.update(torch.matmul(post_pred,1-batch["mask"]), torch.matmul(batch["gt_depth"],1-batch["mask"]))
        print(np.average(np.array(rmse_temp)))
        state = "Inference: subset -- {} | ".format(subset)
        logger.info(state + metrics_meter.suffix)

# def inference(
#         model, test_loaders, metrics, save_dir="", logger=None
# ):

#     model.eval()
#     metrics_meter = AggregatedMeter(metrics, maxlen=20)
#     for subset, loader in test_loaders.items():
#         idx = 0
#         logger.info(
#             "Inference: subset -- {}. Total number of batches: {}.".format(subset, len(loader))
#         )

#         metrics_meter.reset()
#         # loop over dataset
#         for batch in tqdm(loader):
#             batch = model.preprocess(batch)
#             pred = model(batch)

#             with torch.no_grad():
#                 post_pred = model.postprocess(pred)
#                 if save_dir:
#                     B = batch["color"].shape[0]
#                     for it in range(B):
#                         fig = visualize.figure(
#                             batch["color"][it], batch["raw_depth"][it],
#                             batch["mask"][it], batch["gt_depth"][it],
#                             post_pred[it], close=True
#                         )
#                         fig.savefig(
#                             os.path.join(save_dir, "result_{}.png".format(idx)), dpi=fig.dpi
#                         )

#                         idx += 1

#                 metrics_meter.update(post_pred, batch["gt_depth"])

#         state = "Inference: subset -- {} | ".format(subset)
#         logger.info(state + metrics_meter.suffix)