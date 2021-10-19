import numpy as np
import open3d as o3d

if __name__ == '__main__':
    dset = 'frankascanv2/test'
    name = '1622410107.5037992'
    obj_idx = 0

    transp = o3d.io.read_point_cloud("./%s/%s/depth2pcd_pred_%d.pcd" % (dset, name, obj_idx))
    # print(np.array(transp.points).shape)
    # o3d.visualization.draw_geometries([transp])

    # opaque = o3d.io.read_point_cloud("./%s/%s/depth2pcd_GT_%d.pcd" % (dset, name, obj_idx))
    # print(np.array(opaque.points).shape)
    # o3d.visualization.draw_geometries([opaque])

    opaque = o3d.io.read_point_cloud("./%s/%s/Ground_Truth_recenter_%d.pcd" % (dset, name, obj_idx))
    print(np.array(opaque.points).shape)
    o3d.visualization.draw_geometries([opaque])

    pred_i = o3d.io.read_point_cloud("./%s/%s/depth2pcd_pred_%d.pcd" % (dset, name, obj_idx))
    print(np.array(pred_i.points).shape)
    o3d.visualization.draw_geometries([pred_i])

    # depth_raw = IO.get("./%s/%s/depth.exr" % (dset, name))
    # plt.imshow(depth_raw)
    # plt.show()
    # pt_raw = deproject(depth_raw, INV_K)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pt_raw)
    # print(np.array(pcd.points).shape)
    # o3d.visualization.draw_geometries([pcd])
    #
    # depth_gt = IO.get("./%s/%s/detph_GroundTruth.exr" % (dset, name))
    # plt.imshow(depth_gt)
    # plt.show()
    # pt_gt = deproject(depth_gt, INV_K)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pt_gt)
    # print(np.array(pcd.points).shape)
    # o3d.visualization.draw_geometries([pcd])
    #
    # # Visualize groundtruth point clouds of all objects and the background
    # mask = IO.get("%s/%s/instance_segment.png" % (dset, name))
    # depth_gt = IO.get("./%s/%s/detph_GroundTruth.exr" % (dset, name))
    # mask_bg = np.array(mask[:, :, 2] == 0, dtype=np.float32)
    # depth_bg = depth_gt * mask_bg
    # pt = deproject(depth_bg, INV_K)
    # mask_vals = np.unique(mask[:, :, 2])[1:]
    # for i in range(len(mask_vals)):
    #     pcd_gt_i = IO.get("%s/%s/Ground_Truth_%d.pcd" % (dset, name, i))
    #     pt = np.concatenate((pt, pcd_gt_i), axis=0)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pt)
    # print(np.array(pcd.points).shape)
    # o3d.visualization.draw_geometries([pcd])
    #
    # depth_pred = IO.get("./%s/%s/depth_pred.exr" % (dset, name))
    # plt.imshow(depth_pred)
    # plt.show()
    # pt_pred = deproject(depth_pred, INV_K)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pt_pred)
    # print(np.array(pcd.points).shape)
    # o3d.visualization.draw_geometries([pcd])
