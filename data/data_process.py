import os
import cv2
import json
import numpy as np
from PIL import Image
import random


def auto_increment_integer_generator():
    i = 1
    while True:
        yield i
        i += 1


if __name__ == '__main__':
    path = './ann/2/MyEllipse_json_2.json'
    text = json.load(open(path, 'r'))

    image_id_generator = auto_increment_integer_generator()
    annotation_id_generator = auto_increment_integer_generator()
    images_list = []
    annotation_list = []

    # test_sample = random.sample(range(0, 100), 10)
    # val_sample = random.sample(test_sample, 5)
    # while len(val_sample) < 10:
    #     temp = random.choice(range(0, 100))
    #     if temp not in test_sample and temp not in val_sample:
    #         val_sample.append(temp)
    # print(test_sample)
    # print(val_sample)

    test_sample = [8, 34, 23, 30, 89, 13, 94, 43, 98, 84]
    val_sample = [34, 89, 8, 30, 84, 26, 51, 87, 32, 76]


    for x in text:
        img_name = text[x]['filename']
        pil = Image.open(os.path.join('image', img_name))
        width, height = pil.size
        del pil

        if True:
            print(img_name)
            images_list.append(
                {
                    "license": 1,
                    "file_name": os.path.join(img_name),
                    "height": height,
                    "width": width,
                    "id": next(image_id_generator)
                }
            )

            anns = text[x]['regions']
            bboxs = []
            for ann in anns:
                bbox = []
                bbox.append(ann['shape_attributes']['cx'])
                bbox.append(ann['shape_attributes']['cy'])

                a = ann['shape_attributes']['rx']
                b = ann['shape_attributes']['ry']
                theta = ann['shape_attributes']['theta']
                theta = np.rad2deg(theta)

                if a < b:
                    a, b = b, a
                    theta += 90
                while theta > 90 or theta < -90:
                    if theta > 90:
                        theta -= 180
                    if theta < -90:
                        theta += 180
                bbox.append(a)
                bbox.append(b)
                bbox.append(theta)
                image = images_list[-1]
                annotation_list.append(
                    {
                        "iscrowd": 0,
                        "image_id": image.get('id'),
                        "bbox": bbox,
                        "category_id": 1,
                        "id": next(annotation_id_generator)
                    }
                )

    with open('./base.json', 'r') as f:
        original = json.load(f)

    original['images'].extend(images_list)
    original['annotations'].extend(annotation_list)
    print(len(images_list))

    with open('./ann/2/all.json', 'w') as f:
        f.write(json.dumps(original, indent=4))
