# pig-instance-segmentation
## Description
This is the implementation code of our paper named **Dual Attention-guided Feature Pyramid Network for Instance Segmentation of Group Pigs**
  
In this paper, inspire by [DANet](https://arxiv.org/pdf/1809.02983.pdf) and [CCNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_CCNet_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.pdf). We introduce the channel and spatial attention to the Feature Pyramid Network for the instance segmentation of group pigs. We explored the performance difference between these two types of attention and the other existing attentions. At the same time, we discussed the introduction of different levels of spatial attention, which proved the practicality of the two types of attention information in the segmentation of live pigs in different scenarios.

You can see the predict demo video in the folder **img/1.mp4**.
## Install Dependencies
```
conda create -n env_pig
conda activate env_pig
pip install torch==1.3.1
pip install cython
pip install torchvision==0.4.2
pip install albumentations
pip install imagecorruptions
pip install pycocotools
pip install terminaltables
pip install mmcv==0.4.2
sudo pip install -e .
```
When you see **Successfully installed mmdet**, The basic environment is installed successfully！
## Train
Take Mask R-CNN-R50 as example,you should cd the project root path, latter execute the following command
```
  python tools/train.py configs/mask_rcnn_r50_fpn_1x.py --work_dir logs_pig
```
The configs folder includes 8 model configuration files as follows:
- mask_rcnn_r50_fpn_1x.py
- mask_rcnn_r101_fpn_1x.py
- cascade_mask_rcnn_r50_fpn_1x.py
- cascade_mask_rcnn_r101_fpn_1x.py
- ms_rcnn_r50_fpn_1x.py
- ms_rcnn_r101_fpn_1x.py
- htc_without_semantic_r50_fpn_1x.py
- htc_without_semantic_r101_fpn_1x.py

When you want to train your own model, you need to modify the configuration file, take mask_rcnn_r50_fpn_1x.py as example
- Change **num_classes** to the number of categories in your training set + 1
- Search **dataset settings** to change the datasets information 
- Modify the **epoch** or other information according to your own needs  
## Inference
  For inference, you should execute follow command
```
  python inference.py
```

You should modify the values at the beginning of the inference.py file to suit your scenario.
- Change the config_path to the config path.
- Change the model_path to your model .pth path.
- Change the img_folder to the test image floder.
- Change the img_save to the save output path.

Besides, if you want to predict the video, you can also find the code in the inference.py.
## Attention Blocks
- **CBAM:** at the path of mmdet\models\attentions\cbam_attention.py
- **BAM:** at the path of mmdet\models\attentions\bam_attention.py
- **SCSE:** at the path of mmdet\models\attentions\scse_attention.py
- **DAB:** at the path of mmdet\models\attentions\cs_attention.py
- **ACU:** the Asymmetric Convolution Unit can find at the file mmdet\models\necks\fpn.py

## Konwledgement
Our code is developed based on mmdetection framework. In the attention module part, we refer to the following link content, thank the authors for their hard work.
- https://github.com/open-mmlab/mmdetection
- https://github.com/junfu1115/DANet
- https://github.com/speedinghzl/CCNet
- https://github.com/asdf2kr/BAM-CBAM-pytorch/blob/master/Models/attention.py
- https://github.com/DingXiaoH/ACNet/blob/master/acnet/acnet_builder.py


