from logging import root
import os
import torch

import numpy as np
np.random.seed(0)
from PIL import Image
import OpenEXR
import Imath
import json


# ROOT = "/h/helen/datasets/frankascan-recenter"
ROOT = "/h/helen/datasets/frankascan_v2"
def remove_tag(image, tag_mask):
    return 0
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
    # mask[np.where(image <= 0.01)] = 1
    mask[np.where(image <= 0.01)] = 1
    return mask

def png_loader_tag(path_to_png):
    image = np.array(Image.open(path_to_png).convert('L')) / 255. # .transpose([2, 0, 1])
    mask = np.zeros_like(image)
    # mask[np.where(image <= 0.01)] = 1
    mask[np.where(image <= 0.5)] = 1
    return mask


class FrankaScan:
    def __init__(
            self, root=ROOT, split="train", transforms=None, processed=True, model_segment=True,num_vessels = 1, pcc_trans10k = False, remove_tag = False
    ):
        # Determine names of files with the specified vessel number
        # Opening JSON file
        f = open(os.path.join(ROOT,'data.json'))

        # returns JSON object as 
        # a dictionary
        data = json.load(f)

        # Iterating through the json
        # list
        # for i in data['emp_details']:
        #     print(i)
        output = data.copy()

        # Closing file
        f.close()
        # Load json file
        # a_file = open(os.path.join(ROOT,"data.json"), "r")

        # output = a_file.read()

        # # print(output)

        # # a_file.close()
        # print('output type', type(output))
        # print(type(output))
        if split != 'test_val':
            self.vessel_names = output[str(num_vessels)]
        else:
            self.vessel_names = []
            for vessel_num in output.keys():
                self.vessel_names+=output[vessel_num]
        # print('here',type(self.vessel_names), len(self.vessel_names), self.vessel_names)

        # print('ROOT', root)
        self.transforms = transforms
        if split in ['val', 'test', 'test_val']:

            self.data_root = [os.path.join(root,'val'), os.path.join(root,'test')]
             # os.path.join(root, "data")
        
        elif split =='train':
            self.data_root = [os.path.join(root,'train')]
        # self.split_file = os.path.join(root, "splits", split + ".txt")
        # self.data_list = self._get_data_list(self.split_file)
        self.remove_tag = remove_tag
        self.pcc_trans10k = pcc_trans10k
        self.color_name, self.depth_name, self.render_name = [], [], []
        self.normal_name = []
        self.processed = processed
        self.model_segment = model_segment
        self._load_data(processed=processed, pcc_trans10k = pcc_trans10k)

    def _load_data(self,processed=False, pcc_trans10k = False):
        for root_folder in self.data_root:
            print(root_folder)
            for x in os.listdir(root_folder):
                x = str(x)
                scene = os.path.join(root_folder,x)
                if x not in self.vessel_names:
                    # print('Skipping')
                    continue
                if processed:
                    raw_depth_f = os.path.join(scene, 'depth_pred.exr')
                elif pcc_trans10k:
                    raw_depth_f = os.path.join(scene, 'depth_pred_10k.exr')
                else:
                    raw_depth_f = os.path.join(scene, 'depth.exr')
                render_depth_f  = os.path.join(scene, 'detph_GroundTruth.exr')
                color_f = os.path.join(scene, 'image.jpg')

                self.depth_name.append(raw_depth_f)
                self.render_name.append(render_depth_f)
                self.color_name.append(color_f)
        print(len(self.color_name))
            # self.normal_name.append(est_normal_f)

        # for x in os.listdir(self.data_root):
        #     scene               = os.path.join(self.data_root, x)
        #     raw_depth_scene     = os.path.join(scene, 'undistorted_depth_images')
        #     render_depth_scene  = os.path.join(scene, 'render_depth')

        #     for y in os.listdir(raw_depth_scene):
        #         valid, resize_count, one_scene_name, num_1, num_2, png = self._split_matterport_path(y)
        #         if valid == False or png != 'png' or resize_count != 1:
        #             continue
        #         data_id = (x, one_scene_name, num_1, num_2)
        #         if data_id not in self.data_list:
        #             continue
        #         raw_depth_f     = os.path.join(raw_depth_scene, y)
        #         render_depth_f  = os.path.join(render_depth_scene, y.split('.')[0] + '_mesh_depth.png')
        #         color_f         = os.path.join(
        #             scene,'undistorted_color_images', f'resize_{one_scene_name}_i{num_1}_{num_2}.jpg'
        #         )
        #         est_normal_f = os.path.join(
        #             scene, 'estimate_normal', f'resize_{one_scene_name}_d{num_1}_{num_2}_normal_est.png'
        #         )


        #         self.depth_name.append(raw_depth_f)
        #         self.render_name.append(render_depth_f)
        #         self.color_name.append(color_f)
        #         self.normal_name.append(est_normal_f)

    # def _get_data_list(self, filename):
    #     with open(filename, 'r') as f:
    #         content = f.read().splitlines()
    #     data_list = []
    #     for ele in content:
    #         left, _, right = ele.split('/')
    #         valid, resize_count, one_scene_name, num_1, num_2, png = self._split_matterport_path(right)
    #         if valid == False:
    #             print(f'Invalid data_id in datalist: {ele}')
    #         data_list.append((left, one_scene_name, num_1, num_2))
    #     return set(data_list)

    # def _split_matterport_path(self, path):
    #     try:
    #         left, png = path.split('.')
    #         lefts = left.split('_')
    #         resize_count = left.count('resize')
    #         one_scene_name = lefts[resize_count]
    #         num_1 = lefts[resize_count+1][-1]
    #         num_2 = lefts[resize_count+2]
    #         return True, resize_count, one_scene_name, num_1, num_2, png
    #     except Exception as e:
    #         print(e)
    #         return False, None, None, None, None, None

    def __len__(self):
        return len(self.depth_name)

    def __getitem__(self, index):
        color           = np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255. # exr_loader(self.color_name[index], ndim=3) / 255.  #np.array(Image.open(self.color_name[index])).transpose([2, 0, 1]) / 255.
        render_depth    = exr_loader(self.render_name[index], ndim=1) # np.array(Image.open(self.render_name[index])) / 4000.
        depth           = exr_loader(self.depth_name[index], ndim=1) #np.array(Image.open(self.depth_name[index])) / 4000.

        
        # normals = np.array(Image.open(self.normal_name[index])).transpose([2, 0, 1])
        # normals = (normals - 90.) / 180.
        # if self.processed:
        #     mask = np.zeros_like(depth)
        #     mask[np.where(depth > 0)] = 1
        # elif not self.processed:
        #     path, _ = os.path.split(self.depth_name[index])
        #     if self.model_segment:
        #         path_to_mask = os.path.join(path,'image_glass_trans10k.png')
        #     elif not self.model_segment:
        #         path_to_mask = os.path.join(path,'instance_segment.png')
        #     mask = png_loader(path_to_mask)
        # if self.processed:
        #     mask = np.zeros_like(depth)
        #     mask[np.where(depth > 0)] = 1
        # elif not self.processed:
        path, _ = os.path.split(self.depth_name[index])
        if self.model_segment:
            path_to_mask = os.path.join(path,'image_glass_trans10k.png')
            path_to_gt_mask = os.path.join(path,'instance_segment.png')
            mask = png_loader(path_to_mask)
            gt_mask = png_loader(path_to_gt_mask)
        elif not self.model_segment:
            if self.pcc_trans10k or self.processed:
                mask = np.zeros_like(depth)
                mask[np.where(depth > 0)] = 1
                path_to_gt_mask = os.path.join(path,'instance_segment.png')
                gt_mask = png_loader(path_to_gt_mask)
                # path_to_mask = os.path.join(path,'instance_segment.png')
            else:
                path_to_mask = os.path.join(path,'instance_segment.png')
                path_to_gt_mask = os.path.join(path,'instance_segment.png')
                mask = png_loader(path_to_mask)
                gt_mask = mask

        
        render_depth[np.isnan(render_depth)] = 0.0
        render_depth[np.isinf(render_depth)] = 0.0
        depth[np.isnan(depth)] = 0.0
        depth[np.isinf(depth)] = 0.0

        if self.remove_tag:
            tag_path, _ = os.path.split(self.color_name[index])
            tag_path = os.path.join(tag_path, 'tag_mask.png')
            try:
                tag_mask = png_loader_tag(tag_path)
            except:
                print(f'No Tag Mask in Folder {tag_path}')
                tag_mask = np.ones(gt_mask.shape)
            tag_areas_to_remove = ~tag_mask.astype(bool) ^ (np.multiply(~tag_mask.astype(bool),~gt_mask.astype(bool)))

            # Form a random matrix
            # random_matrix = np.random.rand(color.shape[0], color.shape[1], color.shape[2])
            random_matrix = np.ones(color.shape)/2.0
            # color[:,tag_areas_to_remove] = random_matrix[:,tag_areas_to_remove]
            # print(tag_areas_to_remove.shape)
            color[:,tag_areas_to_remove] = random_matrix[:,tag_areas_to_remove]
            # tag_mask = np.array(Image.open(self.color_name[index]).convert('L'))
            # color = remove_tag(color,)

            # Make a numpy array of random values

        # print('max_pixel_depth', np.amax(depth))
        # print('max_pixel_gt', np.amax(render_depth))

        return  {
            'color':        torch.tensor(color, dtype=torch.float32),
            'raw_depth':    torch.tensor(depth, dtype=torch.float32).unsqueeze(0),
            'mask':         torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'gt_mask': torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0),
            #'normals':      torch.tensor(normals, dtype=torch.float32).unsqueeze(0),
            'gt_depth':     torch.tensor(render_depth, dtype=torch.float32).unsqueeze(0),
        }