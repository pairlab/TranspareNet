import os
import numpy as np
from tqdm import tqdm
import json
import cv2
import OpenEXR
import Imath
from grnet_point_cloud_completion.utils.io import IO


# Paths to depth-image data
TRAIN_PATH = "/home/sf3203tr4/Downloads/cup_0/data/cup_0"
TEST_PATH = "./frankascanv2/test"
SAVE_TRAIN_PATH = "/home/sf3203tr4/Downloads/cup_0/data/cup_0"
SAVE_TEST_PATH = "./frankascanv2/test"

# Camera intrinsics (read from camera)
K = np.array([[613.96246338,            0, 324.44714355],
              [           0, 613.75634766, 239.17121887],
              [           0,            0,            1]])
# K = np.array([[675.61713,            0, 632.1181],
#               [           0, 675.61713, 338.28537],
#               [           0,            0,            1]])
INV_K = np.linalg.inv(K)

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
            try:
                channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
            except:
                channel = np.frombuffer(exr_file.channel('D', pt), dtype=np.float32)
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

def deproject(depth_image, inv_k):
    """Deprojects a DepthImage into a PointCloud.
    Reference: Berkeley AutoLab Core
    https://github.com/BerkeleyAutomation/perception/blob/e1c936f38a0aef97348c2d8de364807b5238e1d0/perception/camera_intrinsics.py#L335
    """
    # create homogeneous pixels
    row_indices = np.arange(depth_image.shape[0])
    col_indices = np.arange(depth_image.shape[1])
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])

    # deproject
    points_3d = depth_arr * inv_k.dot(pixels_homog)
    points_3d = points_3d.transpose()
    return points_3d


def img2pcdkeypose(name, load_path, save_path, inv_k, centering=False, normalize=False, skip=False, plot=False):
    """Convert a pair of ClearGrasp images with opaque and transparent objects into point clouds.
    """
    for j in range(80):
        try:
            k = str(j)
            mask = IO.get(os.path.join(load_path, f'{name}/{k.zfill(6)}_mask.png'))

            # Separate multiple objects into multiple point clouds
            mask_vals = np.unique(mask)[1:]  # each object is indicated by a distinct value in RED channel
            maxdis = []
            skip_count = 0
            for i in range(len(mask_vals)):
                mask_i = np.array(mask == mask_vals[i], dtype=np.float32)
                if plot:
                    cv2.imshow("%s idx:%d" % (name, i), mask_i)
                    cv2.waitKey(0)

                mask_pcd = deproject(mask_i, inv_k).sum(axis=1) > 0
                opaque_depth = exr_loader(os.path.join(load_path, f'{name}/{k.zfill(6)}_Do.exr'))[0]
                opaque_pcd = deproject(opaque_depth, inv_k)[mask_pcd]
                center = opaque_pcd.mean(axis=0)
                transp_depth = exr_loader(os.path.join(load_path, f'{name}/{k.zfill(6)}_Dt.exr'))[0]
                transp_pcd = deproject(transp_depth, inv_k)[mask_pcd]
                if centering:
                    opaque_pcd -= center
                    transp_pcd -= center
                maxdis.append(np.max((np.max(np.abs(opaque_pcd)), np.max(np.abs(transp_pcd)))))
                if normalize and not skip:
                    opaque_pcd /= maxdis[-1] * 1.01
                    transp_pcd /= maxdis[-1] * 1.01
                if maxdis[-1] >= 1 and skip:
                    skip_count += 1
                    continue
                if not os.path.exists(os.path.join(save_path, name)):
                    os.mkdir(os.path.join(save_path, name))
                if maxdis[-1] == 0:
                    print((name, i))
                # save centering and scaling factors
                factors = {
                    'centering': centering,
                    'center_position': center.tolist(),
                    'normalize': normalize,
                    'normalize_factor': maxdis[-1] * 1.01,
                }
                with open(os.path.join(save_path, f'{name}/{k.zfill(6)}_scale.json'), 'w') as outfile:
                    json.dump(factors, outfile)

                IO.put(os.path.join(save_path, f'{name}/{k.zfill(6)}_gt.pcd'), opaque_pcd)
                IO.put(os.path.join(save_path, f'{name}/{k.zfill(6)}_input.pcd'), transp_pcd)
        except:
            continue
        # print(mask_pcd.sum())
        # print(opaque_pcd.shape)
        # print(transp_pcd.shape)
    return np.max(maxdis), skip_count


def img2pcdHelper(mask, depth, inv_k):
    if len(mask.shape) == 3:
        mask = mask[:, :, 2]
    # Separate multiple objects into multiple point clouds
    mask_vals = np.unique(mask)[1:]  # each object is indicated by a distinct value in RED channel
    maxdis = [0]
    pcds = []
    centers = []
    for i in range(len(mask_vals)):
        mask_i = np.array(mask == mask_vals[i], dtype=np.float32)

        mask_pcd = deproject(mask_i, inv_k).sum(axis=1) > 0
        transp_depth = depth
        transp_pcd = deproject(transp_depth, inv_k)[mask_pcd]
        center = transp_pcd.mean(axis=0)
        centers.append(center)
        transp_depth[transp_depth > 1] = 1
        transp_pcd -= center
        maxdis.append(np.max(np.abs(transp_pcd)))
        transp_pcd /= maxdis[-1] * 1.01
        pcds.append(transp_pcd)
    return maxdis, pcds, centers
    

def img2pcd(name, load_path, save_path, inv_k, centering=False, normalize=False, skip=False, plot=False):
    """Convert a pair of ClearGrasp images with opaque and transparent objects into point clouds.
    """
    mask = IO.get(os.path.join(load_path, "%s/instance_segment.png" % name))
    if len(mask.shape) == 3:
        mask = mask[:, :, 2]
    # Separate multiple objects into multiple point clouds
    mask_vals = np.unique(mask)[1:]  # each object is indicated by a distinct value in RED channel
    maxdis = [0]
    skip_count = 0
    for i in range(len(mask_vals)):
        mask_i = np.array(mask == mask_vals[i], dtype=np.float32)
        if plot:
            cv2.imshow("%s idx:%d" % (name, i), mask_i)
            cv2.waitKey(0)

        mask_pcd = deproject(mask_i, inv_k).sum(axis=1) > 0

        opaque_depth = IO.get(os.path.join(load_path, "%s/detph_GroundTruth.exr" % name))
        opaque_pcd = deproject(opaque_depth, inv_k)[mask_pcd]
        center = opaque_pcd.mean(axis=0)
        # opaque_pcd = IO.get(os.path.join(load_path, "%s/Ground_Truth_%d.pcd" % (name, i)))
        opaque_depth[opaque_depth > 1] = 1
        transp_depth = IO.get(os.path.join(load_path, "%s/depth.exr" % name))
        transp_pcd = deproject(transp_depth, inv_k)[mask_pcd]
        # center = transp_pcd.mean(axis=0)
        transp_depth[transp_depth > 1] = 1

        if centering:
            opaque_pcd -= center
            transp_pcd -= center
        maxdis.append(np.max((np.max(np.abs(opaque_pcd)), np.max(np.abs(transp_pcd)))))
        if normalize and not skip:
            opaque_pcd /= maxdis[-1] * 1.01
            transp_pcd /= maxdis[-1] * 1.01
        if maxdis[-1] >= 1 and skip:
            skip_count += 1
            continue
        if not os.path.exists(os.path.join(save_path, name)):
            os.mkdir(os.path.join(save_path, name))
        if maxdis[-1] == 0:
            print((name, i))
        # save centering and scaling factors
        factors = {
            'centering': centering,
            'center_position': center.tolist(),
            'normalize': normalize,
            'normalize_factor': maxdis[-1] * 1.01,
        }
        with open(os.path.join(save_path, "%s/scale_factor_%d.json" % (name, i)), 'w') as outfile:
            json.dump(factors, outfile)

        IO.put(os.path.join(save_path, "%s/depth2pcd_GT_%d.pcd" % (name, i)), opaque_pcd)
        IO.put(os.path.join(save_path, "%s/Ground_Truth_recenter_%d.pcd" % (name, i)), opaque_pcd)
        IO.put(os.path.join(save_path, "%s/depth2pcd_%d.pcd" % (name, i)), transp_pcd)
    return np.max(maxdis), skip_count


if __name__ == '__main__':
    maxdis = []
    skip_count = 0
    centering = False
    normalize = True
    skip = False
    print("Converting point clouds for training set...")
    for subdir, dirs, files in os.walk(TRAIN_PATH):
        for dirname in tqdm(dirs):
            max, skip = img2pcdkeypose(dirname, load_path=TRAIN_PATH, save_path=SAVE_TRAIN_PATH, inv_k=INV_K,
                                centering=centering, normalize=normalize, skip=skip)
    print("Converting point clouds for test set...")
    for subdir, dirs, files in os.walk(TEST_PATH):
        for dirname in tqdm(dirs):
            max, skip = img2pcd(dirname, load_path=TEST_PATH, save_path=SAVE_TEST_PATH, inv_k=INV_K,
                                centering=centering, normalize=normalize, skip=skip)
            maxdis.append(max)
            skip_count += skip
