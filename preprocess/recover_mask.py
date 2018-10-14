import os
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
import numpy as np
import json
import cv2
import random
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def generate_colors(n=1):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

if __name__ == '__main__':
    ann_file = './VIRAT_S_000204_04_000738_000977_F_00001460.json'
    annotations = load_json(ann_file)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    img = []

    for index, ann in enumerate(annotations):
        segmentation = ann['segmentation']
        if type(segmentation['counts']) == list:
            rle = maskUtils.frPyObjects([segmentation], int(segmentation['size'][0]), int(segmentation['size'][1]))
        else:
            rle = [segmentation]
        m = maskUtils.decode(rle)
        color_mask = generate_colors()[0]
        colored_img = np.asarray([m * color_mask[2], m * color_mask[1], m * color_mask[0]]).squeeze()
        colored_img = np.transpose(colored_img, [1, 2, 0])
        # print colored_img.shape
        # assert False
        img.append(colored_img)


    img = np.asarray(sum(img))
    # print img.shape

        # if img.any() == None:
        #     img = np.zeros((m.shape[1], m.shape[0], 3))
        # color_mask = generate_colors()[0]
        # colored_img = np.asarray([m*color_mask[2], m*color_mask[1], m*color_mask[0]])
        # colored_img = np.transpose(colored_img.squeeze(), [2, 1, 0])
        # print img.shape
        # print colored_img.shape
        # assert False

        # img += colored_img
    cv2.imwrite('tmp.jpg', img)


        # color_mask = np.random.random((1, 3)).tolist()[0]
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # print img.shape
        # img = np.dstack((img, m*0.5))
        # print img.shape
        # assert False





        # cv2.imwrite('./{}.jpg'.format(index), m*255)
        # print m.shape
        # print type(m)
        # print np.max(m),np.min(m)
        # assert False
        # img = np.ones((m.shape[0], m.shape[1], 3))
        # # if ann['iscrowd'] == 1:
        # # color_mask = np.array([2.0, 166.0, 101.0]) / 255
        # # if ann['iscrowd'] == 0:
        # color_mask = np.random.random((1, 3)).tolist()[0]
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # img = np.dstack((img, m*0.5))


        # print segment['segmentation'].keys()
        # if type(segment['segmentation']['counts']) == list:
            # rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])


        # print type(segment)

    # print type(data)
    # annotations = COCO(annFile)
    # print annotations

