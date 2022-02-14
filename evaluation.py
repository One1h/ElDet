# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:19:02 2020

@author: Lim
"""
import os
import cv2
import time
import torch
import numpy as np
from backbone.dlanet_dcn import DlaNet
from predict import pre_process, ctdet_decode, post_process, merge_outputs
import pycocotools.coco as coco
from predict import process
from dataset import get_edge


# =============================================================================
# 旋转 IOU
# =============================================================================
def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    #        print(int_area)
    else:
        ious = 0
    return ious


# =============================================================================
# ellipse IOU
# =============================================================================
def iou_ellipse(bbox1, bbox2, shape):
    board1 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(board1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), int(bbox1[4]),
                startAngle=0, endAngle=360, color=1, thickness=-1)
    board2 = np.zeros((shape[0], shape[1]), np.uint8)
    cv2.ellipse(board2, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), int(bbox2[4]),
                startAngle=0, endAngle=360, color=1, thickness=-1)
    board = board1 + board2

    inter = len(board[np.where(board > 1)])
    union = len(board[np.where(board > 0)])

    iou = 1.0 * inter / union if union else 0
    return iou


def get_pre_ret(img_path, model, device):
    image = cv2.imread(img_path)
    # image = get_edge(image, 1)

    images, meta = pre_process(image)
    images = images.to(device)
    output = model(images)
    dets = process(output)

    dets = post_process(dets, meta)
    ret = merge_outputs(dets)

    res = np.empty([1, 7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:, 5] > 0.3]
        tmp_c = np.ones(len(tmp_s)) * (i + 1)
        tmp = np.c_[tmp_c, tmp_s]
        res = np.append(res, tmp, axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()
    return res, image.shape


def get_ap(img_path, gt, model, device, iou_thres=0.5):
    gt_num = 0
    tp = []
    gt_close = []

    pre_ret, shape = get_pre_ret(img_path, model, device)
    lab_ret = gt
    gt_num += len(lab_ret)

    for class_name, x, y, a, b, ang, prob in pre_ret:
        pre_one = np.array([x, y, a, b, ang])
        flag = False
        for lab in lab_ret:
            x_l, y_l, a_l, b_l, ang_l = lab
            # x_l, y_l, ang_l, a_l, b_l = lab # FDDB

            lab_one = np.array([x_l, y_l, a_l, b_l, ang_l])
            iou = iou_ellipse(pre_one, lab_one, shape)
            if lab not in gt_close and iou > iou_thres:
                gt_close.append(lab)
                tp.append([1, prob])
                flag = True
                break
        if not flag:
            tp.append([-1, prob])

    return tp, gt_num


def in_range(x):
    while x > 90:
        x -= 180
    while x < -90:
        x += 180
    return x


def get_ap_theta(img_path, gt, model, device, iou_thres=0.5, theta_thres=10):
    gt_num = 0
    tp = []
    gt_close = []

    pre_ret, shape = get_pre_ret(img_path, model, device)
    lab_ret = gt
    gt_num += len(lab_ret)

    for class_name, x, y, a, b, ang, prob in pre_ret:
        pre_one = np.array([x, y, a, b, ang])
        flag = False
        for lab in lab_ret:
            x_l, y_l, a_l, b_l, ang_l = lab
            # x_l, y_l, ang_l, a_l, b_l = lab # FDDB

            lab_one = np.array([x_l, y_l, a_l, b_l, ang_l])
            iou = iou_ellipse(pre_one, lab_one, shape)
            if lab not in gt_close and iou > iou_thres \
                    and (abs(in_range(lab_one[-1] - pre_one[-1])) < theta_thres or pre_one[2] / pre_one[3] < 1.2):
                gt_close.append(lab)
                tp.append([1, prob])
                flag = True
                break
        if not flag:
            tp.append([-1, prob])

    return tp, gt_num


def evaluation(model, device, iou_thresh=0.5, theta_thresh=10):
    model.eval()
    model.cuda()

    data_coco = coco.COCO('./data/random_divide/annotations/test.json')
    # data_coco = coco.COCO('./data/FDDB/FDDB_coco/ellipse/fddb_test.json')
    imgs_id = data_coco.getImgIds()

    # IoU AP
    gt_num = 0
    tps = []
    for index in imgs_id:
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        image_name = os.path.join('./data/random_divide/images', file_name)
        # image_name = os.path.join('./data/FDDB', file_name)
        ann_ids = data_coco.getAnnIds(imgIds=[index])
        anns = data_coco.loadAnns(ids=ann_ids)
        gt = []
        for ann in anns:
            gt.append(ann['bbox'])

        tp, gt_n = get_ap(image_name, gt, model, device, iou_thresh)
        gt_num += gt_n
        tps.extend(tp)

    tps.sort(key=lambda x: x[-1], reverse=True)
    recall, precision = [0], [0]
    t, n = 0, 0
    for tp in tps:
        n += 1
        if tp[0] == 1:
            t += 1
            recall.append(1.0 * t / gt_num)
        else:
            recall.append(recall[-1])
        precision.append(1.0 * t / n)

    curve = {0: 0}
    for i in range(len(recall)):
        if recall[i] not in curve or curve[recall[i]] < precision[i]:
            curve[recall[i]] = precision[i]
    # print('curve: ', curve)

    keys = [key for key in curve]
    keys.sort()
    ap = 0.0
    for i in range(1, len(keys)):
        ap += curve[keys[i]] * (keys[i] - keys[i - 1])
    print('* iou_ap: ', ap)

    gt_num = 0
    tps = []
    # AP_theta
    for index in imgs_id:
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        image_name = os.path.join('./data/random_divide/images', file_name)
        # image_name = os.path.join('./data/FDDB', file_name)
        ann_ids = data_coco.getAnnIds(imgIds=[index])
        anns = data_coco.loadAnns(ids=ann_ids)
        gt = []
        for ann in anns:
            gt.append(ann['bbox'])

        tp, gt_n = get_ap_theta(image_name, gt, model, device, iou_thresh, theta_thresh)
        gt_num += gt_n
        tps.extend(tp)

    tps.sort(key=lambda x: x[-1], reverse=True)
    recall, precision = [0], [0]
    t, n = 0, 0
    for tp in tps:
        n += 1
        if tp[0] == 1:
            t += 1
            recall.append(1.0 * t / gt_num)
        else:
            recall.append(recall[-1])
        precision.append(1.0 * t / n)

    curve = {0: 0}
    for i in range(len(recall)):
        if recall[i] not in curve or curve[recall[i]] < precision[i]:
            curve[recall[i]] = precision[i]
    # print('curve: ', curve)

    keys = [key for key in curve]
    keys.sort()
    ap = 0.0
    for i in range(1, len(keys)):
        ap += curve[keys[i]] * (keys[i] - keys[i - 1])
    print('* theta_ap: ', ap)
    return ap


if __name__ == '__main__':
    model = DlaNet(34)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('./results/train/temp/last.pth'))

    evaluation(model, device, iou_thresh=0.5, theta_thresh=10)
    evaluation(model, device, iou_thresh=0.75, theta_thresh=10)

    # AP = 0
    # for i in range(5):
    #     AP += evaluation(model, device, iou_thresh=0.75, theta_thresh=10-i*2)
    # print('AP* = ', AP/5)
