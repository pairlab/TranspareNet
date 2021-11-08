# Seeing Glass: Joint Point-Cloud and Depth Completion for Transparent Objects 

The basis of many object manipulation algorithms is RGB-D input. Yet, commodity RGB-D sensors can only provide distorted depth maps for a wide range of transparent objects due light refraction and absorption. To tackle the perception challenges posed by transparent objects, we propose TranspareNet, a joint point cloud and depth completion method, with the ability to complete the depth of transparent objects in cluttered and complex scenes, even with partially filled fluid contents within the vessels. To address the shortcomings of existing transparent object data collection schemes in literature, we also propose an automated dataset creation workflow that consists of robot-controlled image collection and vision-based automatic annotation. Through this automated workflow, we created Transparent Object Depth Dataset (TODD), which consists of nearly 15000 RGB-D images. Our experimental evaluation demonstrates that TranspareNet outperforms existing state-of-the-art depth completion methods on multiple datasets, including ClearGrasp, and that it also handles cluttered scenes when trained on TODD. 

Dataset: https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP3/ZJJAJ3

This repository provides:
- Point cloud completion module
- Depth completion module 
- Dataset collection utility tool

## Installation 

```
git clone https://github.com/pairlab/TranspareNet.git
pip install -r requirements.txt
```

## To Run the Code

TranspareNet inference:
```
python inference_transparenet.py 
```

Point cloud completion training
```
python pccRunner.py
```

Convert dataset depth images to objects point clouds
```
python grnet_point_cloud_completion/datasets/img2pcd.py
```

Convert predicted point clouds to sparse depth estimation
```
pyhton grnet_point_cloud_completion/datasets/pcd2img.py
```

Depth completion training
```
python tools/train_franka.py
```

### Automated Dataset Collection

Requirement:
- [frankapy](https://github.com/iamlab-cmu/frankapy-public)
- Franka Emika panda robot
- Intel Realsense D435i camera
- [Intel Realsense ROS](https://github.com/IntelRealSense/realsense-ros)



## Citiation
Our point cloud completion network is based on [GRNet](https://github.com/hzxie/GRNet), and our depth completion network is based on [DMLRN](https://github.com/saic-vul/saic_depth_completion)


