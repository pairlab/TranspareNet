import torch

import argparse
from saic_depth_completion.data.datasets.franka_scan import FrankaScan
# from saic_depth_completion.data.datasets.franka_scan_separate_by_num import FrankaScan
from saic_depth_completion.data.datasets.keypose import KeyPose
from saic_depth_completion.data.datasets.cleargrasp_test_separate import ClearGrasp
from saic_depth_completion.engine.inference import inference
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel

def main():
    parser = argparse.ArgumentParser(description="Some training params.")

    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="arch0", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--save_dir", default="", type=str, help="Save dir for predictions"
    )
    parser.add_argument(
        "--weights", default= '/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-frankav2_grnet_dm_lrn_1/snapshots/snapshot_70.pth' #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-new_raw_frankav2_gtseg_dm_lrn_1/snapshots/snapshot_70.pth'
        

#'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-new_raw_frankav2_gtseg_dm_lrn_1/snapshots/snapshot_70.pth' #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-frankav2_grnet_dm_lrn_1/snapshots/snapshot_78.pth' #'/h/helen/depth-completion/logs/alienware/snapshot_48.pth' #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-cleargrasp_dm_lrn_2/snapshots/snapshot_100.pth'
        , type=str, metavar="FILE", help="path to config file"
    )
    #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-frankav2_grnet_dm_lrn_1/snapshots/snapshot_90.pth'
    #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-new_raw_frankav2_gtseg_dm_lrn_1/snapshots/snapshot_90.pth'
    # 
    #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-new_raw_frankav2_gtseg_dm_lrn_1/snapshots/snapshot_70.pth'
    # '/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-cleargrasp_dm_lrn_2/snapshots/snapshot_100.pth'
    #'/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-cleargrasp_dm_lrn_2/snapshots/snapshot_100.pth'
    # "/h/helen/depth-completion/logs/DM-LRN|SPADE|efficientnet-b4|imagenet||(1.0*LogDepthL1Loss)|lr=0.0001-raw_frankav2_gtseg_dm_lrn_1/snapshots/snapshot_60.pth"
    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()

    snapshoter = Snapshoter(model, logger=logger)
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
        # 'ssim': SSIM(),
    }

    test_datasets = {
        # "test_cleargrasp_real": ClearGrasp(split='test_real', processed = False),
        # "test_cleargrasp_synthetic": ClearGrasp(split='test_synthetic', processed = False),
        # "val_cleargrasp_real": ClearGrasp(split='val_real', processed = False),
        # "val_cleargrasp_synthetic": ClearGrasp(split='val_synthetic', processed = False)
        # "test_frankascanv2_1": FrankaScan(split='test', processed = True, model_segment=False,num_vessels = 1,  pcc_trans10k = False, remove_tag = True),
        # "test_frankascanv2_2": FrankaScan(split='test', processed = True, model_segment=False,num_vessels = 2,  pcc_trans10k = False, remove_tag = True),
        # "test_frankascanv2_3": FrankaScan(split='test', processed = True, model_segment=False,num_vessels = 3,  pcc_trans10k = False, remove_tag = True),
        # "test_frankascanv2_all": FrankaScan(split='test_val', processed = True, model_segment=False,num_vessels = 3,  pcc_trans10k = False, remove_tag = True),
        # "test_frankascanv2": FrankaScan(split='test', processed = False, model_segment=False, grnet_trans10k = True),
        "val_frankascanv2": FrankaScan(split='val', processed = True, model_segment=False, pcc_trans10k = False, remove_tag = False),
        # "cup_0": KeyPose(split='cup_0',processed = False),
        # "train_frankascanv2": FrankaScan(split='train', processed = False, model_segment=False),
        # "test_matterport": Matterport(split="test"),
        # "official_nyu_test": NyuV2Test(split="official_test"),
        #
        # # first
        # "1gr10pv1pd": NyuV2Test(split="1gr10pv1pd"),
        # "1gr10pv2pd": NyuV2Test(split="1gr10pv2pd"),
        # "1gr10pv5pd": NyuV2Test(split="1gr10pv5pd"),
        #
        # "1gr25pv1pd": NyuV2Test(split="1gr25pv1pd"),
        # "1gr25pv2pd": NyuV2Test(split="1gr25pv2pd"),
        # "1gr25pv5pd": NyuV2Test(split="1gr25pv5pd"),
        #
        # "1gr40pv1pd": NyuV2Test(split="1gr40pv1pd"),
        # "1gr40pv2pd": NyuV2Test(split="1gr40pv2pd"),
        # "1gr40pv5pd": NyuV2Test(split="1gr40pv5pd"),
        #
        # #second
        # "4gr10pv1pd": NyuV2Test(split="4gr10pv1pd"),
        # "4gr10pv2pd": NyuV2Test(split="4gr10pv2pd"),
        # "4gr10pv5pd": NyuV2Test(split="4gr10pv5pd"),
        #
        # "4gr25pv1pd": NyuV2Test(split="4gr25pv1pd"),
        # "4gr25pv2pd": NyuV2Test(split="4gr25pv2pd"),
        # "4gr25pv5pd": NyuV2Test(split="4gr25pv5pd"),
        #
        # "4gr40pv1pd": NyuV2Test(split="4gr40pv1pd"),
        # "4gr40pv2pd": NyuV2Test(split="4gr40pv2pd"),
        # "4gr40pv5pd": NyuV2Test(split="4gr40pv5pd"),
        #
        # # third
        # "8gr10pv1pd": NyuV2Test(split="8gr10pv1pd"),
        # "8gr10pv2pd": NyuV2Test(split="8gr10pv2pd"),
        # "8gr10pv5pd": NyuV2Test(split="8gr10pv5pd"),
        #
        # "8gr25pv1pd": NyuV2Test(split="8gr25pv1pd"),
        # "8gr25pv2pd": NyuV2Test(split="8gr25pv2pd"),
        # "8gr25pv5pd": NyuV2Test(split="8gr25pv5pd"),
        #
        # "8gr40pv1pd": NyuV2Test(split="8gr40pv1pd"),
        # "8gr40pv2pd": NyuV2Test(split="8gr40pv2pd"),
        # "8gr40pv5pd": NyuV2Test(split="8gr40pv5pd"),

    }
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

    inference(
        model,
        test_loaders,
        save_dir=args.save_dir,
        logger=logger,
        metrics=metrics,
        file_names = test_datasets,
        prefix = 'transpareNet',
        visualize_pics = True
    )
