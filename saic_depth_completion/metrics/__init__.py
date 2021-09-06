from .relative import *
from .absolute import *
from .relative_test import *
from .absolute_test import *

from saic_depth_completion.utils.registry import Registry

LOSSES = Registry()

LOSSES["DepthL2Loss"]       = DepthL2Loss
# LOSSES["DepthLogL2Loss"]    = DepthLogL2Loss
LOSSES["LogDepthL1Loss"]    = LogDepthL1Loss
LOSSES["DepthL1Loss"]       = DepthL1Loss
# LOSSES["DepthLogL1Loss"]    = DepthLogL1Loss
LOSSES["SSIM"]              = SSIM
LOSSES["BerHuLoss"]         = BerHuLoss

LOSSES["DepthL2Loss_test"]       = DepthL2Loss_test
# LOSSES["DepthLogL2Loss"]    = DepthLogL2Loss
LOSSES["LogDepthL1Loss_test"]    = LogDepthL1Loss_test
LOSSES["DepthL1Loss_test"]       = DepthL1Loss_test
# LOSSES["DepthLogL1Loss"]    = DepthLogL1Loss
LOSSES["SSIM_test"]              = SSIM_test
LOSSES["BerHuLoss_test"]         = BerHuLoss_test

