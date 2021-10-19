
from configs.grnet.config import cfg
from grnet_point_cloud_completion.core.inferencePCC import inference_pcc
from grnet_point_cloud_completion.models.grnet import GRNet
from functools import partial
import os
import numpy as np
import torch

import argparse
from saic_depth_completion.data.datasets.franka_scan import FrankaScan
from saic_depth_completion.engine.inference import inference
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, DepthL2Loss, DepthL1Loss, DepthRel

def main():
    parser = argparse.ArgumentParser(description='The argument parser of R2Net runner')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    parser.add_argument('--pccweights', dest='pccweights', help='Initialize network from the weights file', default="checkpoints/frankav2-ckpt-epoch-100.pth")
    parser.add_argument('--savepred', dest='save_pred', help='Save predicted point clouds', action='store_true')
    parser.add_argument( "--default_cfg", default="DM-LRN", dest="default_cfg", type=str, help="Default config")
    parser.add_argument( "--config_file", default="configs/dm_lrn/DM-LRN_efficientnet-b4_pepper.yaml", type=str, metavar="FILE", help="path to config file")
    parser.add_argument("--save_dir", default="", type=str, help="Save dir for predictions")
    parser.add_argument("--weights", default='checkpoints/dm-lrn_b4.pth' , type=str, metavar="FILE", help="path to config file")
    args = parser.parse_args()
    dcc_cfg = get_default_config(args.default_cfg)
    dcc_cfg.merge_from_file(args.config_file)
    dcc_cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dccModel = MetaModel(dcc_cfg, device)

    logger = setup_logger()

    snapshoter = Snapshoter(dccModel, logger=logger)
    snapshoter.load(args.weights)

    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
    }

    test_datasets = {"val_frankascanv2": FrankaScan(split='val', processed = True, model_segment=False, pcc_trans10k = False, remove_tag = False),}
    test_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate
        )
        for k, v in test_datasets.items()
    }
    print(test_loaders)

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if args.pccweights is not None:
        cfg.CONST.WEIGHTS = args.pccweights
    if args.save_pred is not None:
        cfg.TEST.SAVE_PRED = args.save_pred
    
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
    torch.backends.cudnn.benchmark = True
    pccModel = GRNet(cfg)
    if torch.cuda.is_available():
        pccModel = torch.nn.DataParallel(pccModel).cuda()
    print(cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    pccModel.load_state_dict(checkpoint['grnet'])
    pccModel.eval()

    K = np.array([[613.96246338,            0, 324.44714355],
              [           0, 613.75634766, 239.17121887],
              [           0,            0,            1]])
    pccPred = partial(inference_pcc, pccModel, cfg.CONST.N_INPUT_POINTS, K)
    
    inference(
        dccModel,
        test_loaders,
        pccPred = pccPred,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
        file_names = test_datasets,
        prefix = 'transpareNet',
        visualize_pics = True
    )

if __name__ == "__main__":
    main()
