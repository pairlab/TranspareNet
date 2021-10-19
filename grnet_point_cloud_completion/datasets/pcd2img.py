import os
import numpy as np
from tqdm import tqdm
import json
import pyexr

from grnet_point_cloud_completion.utils.io import IO
from shutil import copy



TEST_PATH = "./frankascanv2/trainall"
SAVE_TEST_PATH = "frankascanv2/train"
FACTOR_PATH = "./frankascanv2/test"


def project_to_image(k, point_cloud, round_px=True):
    width = 640
    height = 480
    points_proj = k.dot(point_cloud.transpose())
    if len(points_proj.shape) == 1:
        points_proj = points_proj[:, np.newaxis]
    point_depths = points_proj[2, :]
    point_z = np.tile(point_depths, [3, 1])
    points_proj = np.divide(points_proj, point_z)
    if round_px:
        points_proj = np.round(points_proj)
    points_proj = points_proj[:2, :].astype(np.int16)

    valid_ind = np.where((points_proj[0, :] >= 0) & \
                         (points_proj[1, :] >= 0) & \
                         (points_proj[0, :] < width) & \
                         (points_proj[1, :] < height))[0]

    depth_data = np.zeros([height, width])
    points_proj = points_proj[:, valid_ind]
    point_depths = point_depths[valid_ind]
    for i in range(point_depths.shape[0]):
        current_depth = depth_data[points_proj[1,i], points_proj[0,i]]
        point_depth = point_depths[i]
        if current_depth == 0 or point_depth < current_depth:
            depth_data[points_proj[1,i], points_proj[0,i]] = point_depth
    return depth_data

def pcd2imgHelper(mask, depth, k, pcds, maxdis, centers):
    if len(mask.shape) == 3:
        mask = mask[:,:, 2]
    mask_bg = np.array(mask == 0, dtype=np.float32)
    mask_out = np.zeros_like(mask_bg)
    mask_vals = np.unique(mask)[1:]
    depth_in = depth
    depth_out = depth_in * 0
    for i in range(len(mask_vals)):
        pcd_i = pcds[i]
        pcd_i *= maxdis[i] * 1.01
        pcd_i += centers[i]
        depth_i = project_to_image(k, pcd_i)
        mask_i = np.array(mask == mask_vals[i], dtype=np.float32)
        mask_out += mask_i
        depth_out += depth_i * mask_i
    depth_out += depth_in * (mask_out == 0)
    return depth_out


def pcd2img(name, load_path, save_path, k, factor_path=None):
    """Convert point clouds to depth images.
    """
    mask = IO.get(os.path.join(load_path, "%s/instance_segment.png" % name))
    if len(mask.shape) == 3:
        mask = mask[:,:, 2]
    mask_bg = np.array(mask == 0, dtype=np.float32)
    depth_in = IO.get(os.path.join(load_path, "%s/depth.exr" % name))
    depth_out = depth_in * 0
    mask_out = np.zeros_like(mask_bg)
    mask_vals = np.unique(mask)[1:]
    for i in range(len(mask_vals)):
        pcd_i = IO.get(os.path.join(load_path, "%s/depth2pcd_pred_%d.pcd" % (name, i)))
        if factor_path is not None:
            with open(os.path.join(factor_path, "%s/scale_factor_%d.json" % (name, i))) as f:
                factors = json.loads(f.read())
                if factors['normalize']:
                    pcd_i *= float(factors['normalize_factor'])
                if factors['centering']:
                    pcd_i += np.array(factors['center_position'])
        depth_i = project_to_image(k, pcd_i)
        mask_i = np.array(mask == mask_vals[i], dtype=np.float32)
        mask_out += mask_i
        depth_out += depth_i * mask_i
    depth_out += depth_in * (mask_out == 0)
    # write predicted depth image in exr
    depth_out = np.expand_dims(depth_out, -1).repeat(3, axis=2)
    # os.mkdir(os.path.join(save_path, name))
    pyexr.write(os.path.join(save_path, "%s/depth_pred.exr" % name), depth_out)


if __name__ == '__main__':
    print("Converting point clouds to depth images...")
    for subdir, dirs, files in os.walk(TEST_PATH):
        for dirname in tqdm(sorted(dirs)):
            # try:
            #     pcd2img(dirname, load_path=TEST_PATH, save_path=SAVE_TEST_PATH, k=K, factor_path=FACTOR_PATH)
            # except:
            #     print(dirname)
            old_path = os.path.join(TEST_PATH, dirname)
            path = os.path.join(SAVE_TEST_PATH, dirname)

            if not os.path.isdir(path):
                os.makedirs(path)
            try:
                copy(os.path.join(old_path, "depth.exr"), path)
                copy(os.path.join(old_path, "pose_type.json"), path)
                # copy(os.path.join(old_path, "tag_mask.png"), path)
            except:
                print(dirname)
                continue
            try:
                copy(os.path.join(old_path, "apriltag.pkl"), path)
            except:
                print(dirname)

            copy(os.path.join(old_path, "detph_GroundTruth.exr"), path)
            copy(os.path.join(old_path, "image.jpg"), path)
            copy(os.path.join(old_path, "instance_segment.png"), path)


    #pcd2img('1622410107.5037992', TEST_PATH, SAVE_TEST_PATH, K, factor_path=FACTOR_PATH)
