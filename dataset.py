import os
import cv2
import math
import random
import numpy as np
import torch.utils.data as data
import pycocotools.coco as coco


class ctDataset(data.Dataset):
    num_classes = 1
    default_resolution = [512, 512]

    def __init__(self, data_dir='./data', split='train'):
        self.data_dir = data_dir
        self.split = split
        try:
            if split == 'train':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'train.json')
            elif split == 'val':
                self.annot_path = os.path.join(self.data_dir, 'annotations', 'test.json')
        except:
            print('No any data!')

        self.max_objs = 100
        self.class_name = ['obj']
        self._valid_ids = [1]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.split = split
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.data_dir, 'images', file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        keep_res = False  #
        if keep_res:
            input_h = (height | 31) + 1
            input_w = (width | 31) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = 512, 512

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        # Augment
        # inp = grayscale(inp, 0.5)
        inp = get_edge(inp, prob=1)

        inp = (inp.astype(np.float32) / 255.)

        # 归一化
        inp = inp.transpose(2, 0, 1)

        down_ratio = 4
        output_h = input_h // down_ratio
        output_w = input_w // down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        ab = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        mask = np.zeros((num_classes, 128, 128), dtype=np.float32)

        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):
            ann = anns[k]
            bbox = ann['bbox']  # x,y,angle,a,b
            cls_id = int(self.cat_ids[ann['category_id']])

            # 数据扩充和resize之后的变换
            bbox[:2] = affine_transform(bbox[:2], trans_output)

            bbox[2:4] = affine_transform(bbox[2:4], trans_output, bbox[4], mode='ab')
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            bbox[2] = np.clip(bbox[2], 0, output_w - 1)
            bbox[3] = np.clip(bbox[3], 0, output_h - 1)

            a, b, an = bbox[2], bbox[3], bbox[4]
            if a > 0 and b > 0:
                radius = gaussian_radius((math.ceil(b * 2.0), math.ceil(a * 2.0)))
                radius = max(0, int(radius))
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                ab[k] = 1. * a, 1. * b
                ang[k] = 1. * an + 90
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                cv2.ellipse(mask[cls_id], (int(ct[0]), int(ct[1])), (int(ab[k][0]), int(ab[k][1])), int(ang[k][0] - 90),
                            0, 360, 1, -1)

        # inp: 512*512 input | hm: heatmap class | reg_mask: obj data mask | ind: center pixel index
        # wh: width & height | ang: angle
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'ab': ab, 'ang': ang, 'mask': mask}
        reg_offset_flag = True  #
        if reg_offset_flag:
            ret.update({'reg': reg})
        return ret


def get_edge(img, prob=0.5):
    if random.randint(0, 9) in range(0, int(10 * prob)):
        img_b = img[:, :, 0]
        img_g = img[:, :, 1]
        img_r = img[:, :, 2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        img_1 = cv2.Laplacian(img, -1, ksize=5)
        img_1 = cv2.normalize(img_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_2 = cv2.Canny(img, 50, 150)
        img_2 = cv2.normalize(img_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_3 = 0.5 * cv2.Sobel(img, -1, 0, 1, 5) + 0.5 * cv2.Sobel(img, -1, 1, 0, 5)
        img_3 = cv2.normalize(img_3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        img_4 = cv2.normalize(img_4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        img = np.stack([img_b, img_g, img_r, img_1, img_2, img_3, img_4], axis=2)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return img


def grayscale(img, prob=0.5):
    if random.randint(0, 9) in range(0, int(10 * prob)):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t, angle=0, mode='xy'):
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
        new_pt = np.zeros((2, 1), dtype=np.float32)
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)
    return new_pt[:2]


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)
