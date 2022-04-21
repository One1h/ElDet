import os
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from backbone.dlanet_dcn import DlaNet
from Loss import _gather_feat
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat
import pycocotools.coco as coco
from dataset import get_edge



def draw(img, result):
    '''Draw ellipse box'''
    for class_name, x, y, a, b, ang, prob in result:
        print('pred:', [x, y, a, b, ang])
        result = np.array(result)
        x = int(x)
        y = int(y)
        a = int(a)
        b = int(b)
        angle = int(ang)
        # cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (255, 255, 0), 3)
        cv2.ellipse(img, (x, y), (a, b), angle, 0, 360, (0, 0, 255), 3)

    return img


def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

    mean = np.array([0.5194416012442385, 0.5378052387430711, 0.533462090585746],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966],
                   dtype=np.float32).reshape(1, 1, 3)

    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)  # 三维reshape到4维，（1，3，512，512）

    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, ab, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    ab = _transpose_and_gather_feat(ab, inds)
    ab = ab.view(batch, K, 2)

    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)

    # ang = ang.view(batch, K, 8)
    # ang = torch.sigmoid(ang)
    # ang_oct = torch.zeros(batch, K, 1).cuda()
    # oct = angle_label_decode(np.round(torch.sigmoid(ang[0]).cpu().detach().numpy()), 180, 180 / 256., mode=1) * -1
    # ang_oct[0] = torch.Tensor([oct]).cuda().permute(1, 0)
    # ang = ang_oct

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs,
                        ys,
                        ab[..., 0:1],
                        ab[..., 1:2],
                        ang - 90], dim=2)
    # 90 + ang], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def process(output):
    with torch.no_grad():
        hm = output['hm'].sigmoid_()
        # cv2.imshow('', hm[0].cpu().numpy().transpose(1,2,0))
        # cv2.waitKey()
        # ang = output['ang'].relu_()
        ang = output['ang']
        ab = output['ab']
        reg = output['reg']
        # mask = output['mask']
        # cv2.imshow('', mask[0].cpu().numpy().transpose(1,2,0))
        # cv2.waitKey()
        torch.cuda.synchronize()

        dets = ctdet_decode(hm, ab, ang, reg=reg, K=100)  # K 是最多保留几个目标
        return dets


def affine_transform(pt, t, angle=0, mode='xy'):
    new_pt = np.zeros(2, dtype=np.float32)
    if mode == 'xy':
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
    elif mode == 'ab':
        angle = np.deg2rad(angle)
        cosA = np.abs(np.cos(angle))
        sinA = np.abs(np.sin(angle))
        a_x = pt[0] * cosA
        a_y = pt[0] * sinA
        b_x = pt[1] * sinA
        b_y = pt[1] * cosA
        new_pt_a = np.array([a_x, a_y, 0.], dtype=np.float32).T
        new_pt_a = np.dot(t, new_pt_a)
        new_pt_b = np.array([b_x, b_y, 0.], dtype=np.float32).T
        new_pt_b = np.dot(t, new_pt_b)
        new_pt = np.zeros(2, dtype=np.float32)
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size, ang, mode='xy'):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        if mode == 'ab':
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans, ang[p], 'ab')
        else:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans, ang[p])
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h), dets[i, :, 4])
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h), dets[i, :, 4], mode='ab')

        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 1
    dets = ctdet_post_process(dets.copy(),
                              [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
        kth = len(scores) - max_obj_per_img
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2 + 1):
            keep_inds = (detections[j][:, 5] >= thresh)
            detections[j] = detections[j][keep_inds]
    return detections


def predict():
    model = DlaNet(34)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('./results/train/FDDB-iou-15-ab-mask1/best.pth'))
    model.eval()
    model.cuda()

    # data_coco = coco.COCO('./data/random_divide/annotations/test.json')
    data_coco = coco.COCO('./data/FDDB/FDDB_coco/ellipse/fddb_test.json')

    imgs_id = data_coco.getImgIds()
    for index in imgs_id:
        time_beg = time.time()
        file_name = data_coco.loadImgs(ids=[index])[0]['file_name']
        # image_name = os.path.join('./data/random_divide/images', file_name)
        image_name = os.path.join('./data/FDDB', file_name)

        print(image_name)
        image = cv2.imread(image_name)
        img = image
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
        img = draw(img, res)

        time_end = time.time()
        print('time:', time_end-time_beg)

        # ann_ids = data_coco.getAnnIds(imgIds=[index])
        # anns = data_coco.loadAnns(ids=ann_ids)
        # for ann in anns:
        #     bbox = ann['bbox']
        #     print('gt:', bbox)
        #     # print('gt:', (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), bbox[4])
        #     cv2.ellipse(img, (int(bbox[0]), int(bbox[1])),
        #                 (int(bbox[2]), int(bbox[3])),
        #                 int(bbox[4]), 0, 360, (255, 255, 0), 3)
        #
        #     # cv2.ellipse(img, (int(bbox[0]), int(bbox[1])),
        #     #             (int(bbox[3]), int(bbox[4])),
        #     #             int(bbox[2]), 0, 360, (0, 0, 255), 3)   # FDDB

        save_path = os.path.join('./results/predict', file_name.split('/')[-1])
        cv2.imwrite(save_path, img)


def predict_once():
    model = DlaNet(34)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('./results/train/iou-15-ab-mask/best.pth'))
    model.eval()
    model.cuda()

    data_path = './data/test'
    image_name = os.listdir(data_path)

    for name in image_name:
        time_beg = time.time()
        image_name = os.path.join(data_path, name)

        print(image_name)
        image = cv2.imread(image_name)
        img = image
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
        img = draw(img, res)

        time_end = time.time()
        print('time:', time_end-time_beg)

        save_path = os.path.join('./results/test', name.split('/')[-1])
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    # predict()
    predict_once()
