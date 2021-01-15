import os
import numpy as np
import mmcv
import torch
import cv2
import time
import json
import PIL.Image as image
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmdet.core import get_classes
from numpy import mat
import pycocotools.mask as maskutils
import mmdet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config_path = 'configs/pig/cascade_mask_rcnn_r50_fpn_1x.py'
# model_path = 'logs_pig/cascade_mask_rcnn/r50/s_ccattention_3/epoch_5.pth'
# img_folder = 'data/pig/test/image'
# img_save = 'outputs'

config_path = 'configs/mask_rcnn_r50_fpn_1x.py'
model_path = 'logs_pig/epoch_10.pth'
img_folder = 'data/pig/temp'
img_save = 'outputs'

input_video_path = "../data/data28258/9_1.mp4"
output_video_path = "video/9_1_label.mp4"
img_save_path = "1_1.jpg"
video_size = (1920, 1080)
color_list = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,69,0], [41,36,33], [112,128,105], [56,94,15], [160,32,240], [218,165,105], [0,199,140], [252,230,201], [255,215,0]]

def show_mask_result(img, result, save_img, dataset='coco', score_thr=0.7, with_mask=True, fps=0):
    segm_result = None
    if with_mask:
        bbox_result, segm_result = result
    else:
        bbox_result = result
    if isinstance(dataset, str):  # add own data label to mmdet.core.class_name.py
        class_names = get_classes(dataset)
        # print(class_names)
    elif isinstance(dataset, list):
        class_names = dataset
    else:
        raise TypeError('dataset must be a valid dataset name or a list'
                        ' of class names, not {}'.format(type(dataset)))
    h, w, _ = img.shape
    img_show = img[:h, :w, :]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    
    bbox_list = []
    for i in range(len(bboxes)):
        bbox_list.append(bboxes[i][0])
    sorted_nums = sorted(enumerate(bbox_list), key=lambda x: x[1])
    idx = [i[0] for i in sorted_nums]
    nums = [i[1] for i in sorted_nums]
    
    # 改变bbox以及segment的顺序目的是为了能够使得预测的结果中的颜色有序
    temp_bboxes = []
    temp_segm_result = []
    result_segm_result = []
    for i in range(len(idx)):
        temp_bboxes.append(bboxes[idx[i]])
        temp_segm_result.append(segm_result[0][idx[i]])
    result_segm_result.append(temp_segm_result)
    bboxes = temp_bboxes
    bboxes = np.array(bboxes)
    segm_result = result_segm_result
    
    if with_mask:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        count = -1
        for i in inds:
            count = count + 1
            # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            if count >= 15:
                break
            color_mask = mat(color_list)[count]
            print(color_mask)
            mask = maskutils.decode(segms[i]).astype(np.bool)
            img_show[mask] = img_show[mask] * 0.4 + color_mask * 0.6
    
    # only mask(image)
    img = image.fromarray(img_show)
    img.save(save_img)
    # cv2.imwrite(save_img, img_show)
    result_img = img_show
    
    # mask and detection(image)
    # result_img = mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names,
    #                                     score_thr=score_thr, show=False, out_file=save_img, 
    #                                     thickness=2, font_scale=0.8, bbox_color='red', text_color='red')
    # video
    # result_img = mmcv.imshow_det_bboxes(img_show, bboxes, labels, class_names=class_names,
    #                                     score_thr=score_thr, show=False, out_file=save_img, 
    #                                     thickness=2, font_scale=0.8, bbox_color='red', text_color='red',fps=fps)
    return result_img


cfg = mmcv.Config.fromfile(config_path)
cfg.model.pretrained = None
cfg.data.test.test_mode = True

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
model.cfg = cfg
device = torch.device("cuda:0")
model = model.to(device)
model.eval()
model_checkpoint = load_checkpoint(model, model_path)

mask_score_thresh = 0.6

img = []
for single_img in os.listdir(img_folder):
    img.append(os.path.join(img_folder, single_img))

# img_result = inference_detector(model, img)
for single_img in img:
    print(single_img)
    img = mmcv.imread(single_img)
    result = inference_detector(model, img)
    # result = (result[0], result[1][0])
    img_name = os.path.basename(single_img)
    new_path = os.path.join(img_save, img_name)
    show_mask_result(img, result, new_path,
                              score_thr=mask_score_thresh, with_mask=True)

# video
# video = mmcv.VideoReader(input_video_path)
# vid = cv2.VideoCapture(input_video_path)
# video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
# video_fps = vid.get(cv2.CAP_PROP_FPS)
# out_video = cv2.VideoWriter(output_video_path, video_FourCC, video_fps, video_size)
# count = 0
# for frame in video:
#     count = count + 1
#     print(count)
#     start = time.time()
#     result = inference_detector(model, frame)
#     end = time.time()
#     interval = round(end-start, 3)
#     fps = round(1.0 / float(interval), 1)
#     # result = (result[0], result[1][0])
#     result_frame = show_mask_result(frame, result, img_save_path, score_thr=mask_score_thresh, with_mask=True, fps=fps)
#     result_frame = np.asarray(result_frame)
#     result_frame = result_frame.astype(np.uint8)
#     out_video.write(np.uint8(result_frame))