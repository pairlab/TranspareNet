import os
import torch

import numpy as np
np.random.seed(0)
from PIL import Image
import OpenEXR
import Imath
import glob
from skimage.transform import resize

ROOT = "/h/helen/datasets/cleargrasp"

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

def png_loader(path_to_png):
    image = np.array(Image.open(path_to_png).convert('L')) / 255. # .transpose([2, 0, 1])
    mask = np.zeros_like(image)
    mask[np.where(image <= 0.01)] = 1
    return mask


class ClearGrasp:
    def __init__(
            self, root=ROOT, split="train", transforms=None, processed=True
    ):
        # Split can be train or test-val
        self.transforms = transforms
        self.split = split
        if split in ['val','test']:
            split = 'test-val'
        self.data_root = os.path.join(root,'cleargrasp-dataset-'+split) # os.path.join(root, "data")
        # self.split_file = os.path.join(root, "splits", split + ".txt")
        # self.data_list = self._get_data_list(self.split_file)
        self.color_name, self.depth_name, self.render_name,self.mask_name = [], [], [], []
        self.normal_name = []
        self.processed = processed
        self._load_data(processed=processed)
        

    def _load_data(self,processed=True):

        # Check if split is test or validation
        split_type = self.split

        if split_type == 'train':
            models = os.listdir(self.data_root)
            for model in models:
                render_depth_f = sorted(glob.glob(os.path.join(self.data_root,model,'depth-imgs-rectified','*')))
                color_f = sorted(glob.glob(os.path.join(self.data_root,model,'rgb-imgs','*')))
                #sorted(os.listdir(os.path.join(self.data_root,model,'rgb-imgs')),key = lambda x: int(x.split('-')[0]))
                mask_f = sorted(glob.glob(os.path.join(self.data_root,model,'segmentation-masks','*')))
                #sorted(os.listdir(os.path.join(self.data_root,model,'segmentation-masks')),key = lambda x: int(x.split('-')[0])))
                self.render_name += render_depth_f
                self.color_name += color_f
                self.depth_name += render_depth_f
                self.mask_name += mask_f
        
        elif split_type == 'val':
            test_val_folders = os.listdir(self.data_root)
            real_data = ['real-val']
            synthetic_data = ['synthetic-val']

            # List of extensions
            EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg']  #'-rgb.jpg' - includes normals-rgb.jpg
            EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr']
            EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr']
            EXT_MASK = ['-mask.png']

            for folder in test_val_folders:
                if folder in real_data:
                    for subfolder in os.listdir(os.path.join(self.data_root,folder)):
                        for ext in EXT_COLOR_IMG:
                            color_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.color_name += color_f
                        for ext in EXT_DEPTH_IMG:
                            depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.depth_name += depth_f
                        for ext in EXT_DEPTH_GT:
                            render_depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.render_name += render_depth_f
                        for ext in EXT_MASK:
                            mask_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.mask_name += mask_f
                elif folder in synthetic_data:
                    models = os.listdir(os.path.join(self.data_root,folder))
                    for model in models:
                        render_depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'depth-imgs-rectified','*')))
                        color_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'rgb-imgs','*')))
                        #sorted(os.listdir(os.path.join(self.data_root,model,'rgb-imgs')),key = lambda x: int(x.split('-')[0]))
                        mask_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'segmentation-masks','*')))
                        # render_depth_f = sorted(os.listdir(os.path.join(self.data_root,folder,model,'depth-imgs-rectified')),key = lambda x: int(x.split('-')[0]))
                        # color_f = sorted(os.listdir(os.path.join(self.data_root,folder,model,'rgb-imgs')),key = lambda x: int(x.split('-')[0]))
                        # mask_f = sorted(os.listdir(os.path.join(self.data_root,folder,model,'segmentation-masks')),key = lambda x: int(x.split('-')[0]))
                        self.render_name += render_depth_f
                        self.color_name += color_f
                        self.depth_name += render_depth_f
                        self.mask_name += mask_f
        
        elif split_type == 'test':
            test_val_folders = os.listdir(self.data_root)
            real_data = ['real-test']
            synthetic_data = ['synthetic-test']

            # List of extensions
            EXT_COLOR_IMG = ['-transparent-rgb-img.jpg', '-rgb.jpg']  #'-rgb.jpg' - includes normals-rgb.jpg
            EXT_DEPTH_IMG = ['-depth-rectified.exr', '-transparent-depth-img.exr']
            EXT_DEPTH_GT = ['-depth-rectified.exr', '-opaque-depth-img.exr']
            EXT_MASK = ['-mask.png']

            for folder in test_val_folders:
                if folder in real_data:
                    for subfolder in os.listdir(os.path.join(self.data_root,folder)):
                        for ext in EXT_COLOR_IMG:
                            color_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.color_name += color_f
                        for ext in EXT_DEPTH_IMG:
                            depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.depth_name += depth_f
                        for ext in EXT_DEPTH_GT:
                            render_depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.render_name += render_depth_f
                        for ext in EXT_MASK:
                            mask_f = sorted(glob.glob(os.path.join(self.data_root,folder,subfolder,'*' +ext)))
                            self.mask_name += mask_f
                elif folder in synthetic_data:
                    models = os.listdir(os.path.join(self.data_root,folder))
                    for model in models:
                        render_depth_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'depth-imgs-rectified','*')))
                        color_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'rgb-imgs','*')))
                        #sorted(os.listdir(os.path.join(self.data_root,model,'rgb-imgs')),key = lambda x: int(x.split('-')[0]))
                        mask_f = sorted(glob.glob(os.path.join(self.data_root,folder,model,'segmentation-masks','*')))
                        self.render_name += render_depth_f
                        self.color_name += color_f
                        self.depth_name += render_depth_f
                        self.mask_name += mask_f
        
        else:
            raise ValueError('dataloading error, please provide a reasonable split')


    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color           = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255. # exr_loader(self.color_name[index], ndim=3) / 255.  #np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        render_depth    = exr_loader(self.render_name[index], ndim=1) # np.array(Image.open(self.render_name[index])) / 4000.
        depth           = exr_loader(self.depth_name[index], ndim=1) #np.array(Image.open(self.depth_name[index])) / 4000.

        # Load the mask
        mask = png_loader(self.mask_name[index])

        if self.depth_name[index].endswith('depth-rectified.exr'):
            # Remove the portion of the depth image with transparent object
            # If image is synthetic
            depth[np.where(mask==0)] = 0
        
        # Resize arrays:
        
        color =resize(color, (3,240,320))
        assert len(render_depth.shape) == 2 , 'There is channel dimension'
        render_depth = resize(render_depth,(240,320))
        depth =resize(depth,(240,320))
        mask = resize(mask,(240,320))

        render_depth[np.isnan(render_depth)] = 0.0
        render_depth[np.isinf(render_depth)] = 0.0


        return  {
            'color':        torch.tensor(color, dtype=torch.float32),
            'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            #'normals':      torch.tensor(normals, dtype=torch.float32).unsqueeze(0),
            'gt_depth':     torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
        }