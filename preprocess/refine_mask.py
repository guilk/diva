import os
import cv2
import numpy as np
import json
import pycocotools.mask as maskUtils
import random


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def load_vibe_mask(vibe_mask_path):
    vibe_mask = cv2.imread(vibe_mask_path,cv2.IMREAD_GRAYSCALE)
    ret, gray = cv2.threshold(vibe_mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)

    _, contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask = np.ones(gray.shape[:2], np.uint8)*255
    for cnt in contours:
        if cv2.contourArea(cnt) <= 100:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    vibe_mask = cv2.bitwise_and(vibe_mask, vibe_mask, mask=mask)
    # vibe_mask
    # kernel = np.ones((3, 3), np.uint8)
    # vibe_mask = cv2.dilate(img, kernel, iterations=2)

    return vibe_mask


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

def refine_mask(vibe_mask, mrcnn_mask_path):
    annotations = load_json(mrcnn_mask_path)
    img = []

    for index, ann in enumerate(annotations):
        # if ann['cat_name'] not in ['car', 'person', 'truck']:
        #     print ann['cat_name']
        if ann['cat_name'] != 'car' and ann['cat_name'] != 'person':
            continue
        segmentation = ann['segmentation']
        if isinstance(segmentation, list):
            rles = maskUtils.frPyObjects(segmentation, int(segmentation[0]['size'][0]), int(segmentation[0]['size'][1]))
            rle = maskUtils.merge(rles)
        elif type(segmentation['counts']) == list:
            rle = maskUtils.frPyObjects([segmentation], int(segmentation['size'][0]), int(segmentation['size'][1]))
        else:
            rle = [segmentation]
        m = maskUtils.decode(rle).squeeze()
        # print np.max(m), np.min(m)
        # print m.shape[0]*m.shape[1], np.sum(m)
        # assert False
        logical_map = np.logical_and(vibe_mask, m).astype(int)
        if np.sum(logical_map) < 0.1*np.sum(m):
            m = np.zeros_like(m)
        img.append(m)

    mrcnn_mask = np.asarray(sum(img))
    mrcnn_mask[mrcnn_mask >= 1] = 1
    return mrcnn_mask*255

def load_mrcnn_mask(mrcnn_mask_path):
    # mrcnn_mask_path = './VIRAT_S_000207_02_000498_000530/VIRAT_S_000207_02_000498_000530_F_00000449.json'
    annotations = load_json(mrcnn_mask_path)
    img = []

    print len(annotations)

    for index, ann in enumerate(annotations):
        segmentation = ann['segmentation']
        if isinstance(segmentation, list):
            rles = maskUtils.frPyObjects(segmentation, int(segmentation[0]['size'][0]), int(segmentation[0]['size'][1]))
            rle = maskUtils.merge(rles)
        elif type(segmentation['counts']) == list:
            rle = maskUtils.frPyObjects([segmentation], int(segmentation['size'][0]), int(segmentation['size'][1]))
        else:
            rle = [segmentation]
        m = maskUtils.decode(rle)

        color_mask = generate_colors()[0]
        colored_img = np.asarray([m * color_mask[2], m * color_mask[1], m * color_mask[0]]).squeeze()
        colored_img = np.transpose(colored_img, [1, 2, 0])

        img.append(colored_img)

    mrcnn_mask = np.asarray(sum(img))

    cv2.imshow('mrcnn', mrcnn_mask)
    cv2.waitKey(0)
    return mrcnn_mask

def refine_main(video_folder_name):
    vibe_mask_root = '../../../datasets/virat/vibe_masks/'
    # vibe_mask_root = './'
    vibe_mask_folder_path = os.path.join(vibe_mask_root, video_folder_name)

    frame_root = '../../../datasets/virat/frames'
    # frame_root = './'
    frame_folder_path = os.path.join(frame_root, video_folder_name)

    # VIRAT_S_040101_05_000722_001547_F_00004070.json
    mrcnn_mask_root = '../../../datasets/virat/mrcnn_masks/'
    # mrcnn_mask_root = './VIRAT_S_000207_02_000498_000530'

    dst_root = '../../../datasets/virat/refined_masks/'
    dst_folder_path = os.path.join(dst_root, video_folder_name)
    if not os.path.exists(dst_folder_path):
        os.makedirs(dst_folder_path)

    # vibe_masks = os.listdir(vibe_mask_folder_path)
    frames = os.listdir(frame_folder_path)
    frames = sorted(frames)
    # frames = ['00227.jpg']
    for frame_name in frames:
        frame_index = int(frame_name.split('.')[0])
        vibe_mask_path = os.path.join(vibe_mask_folder_path, '{}.jpg'.format(frame_index))
        mrcnn_mask_path = os.path.join(mrcnn_mask_root, '{}_F_{}.json'.format(video_folder_name, str(frame_index).zfill(8)))
        vibe_mask = load_vibe_mask(vibe_mask_path)
        # mrcnn_mask = load_mrcnn_mask(mrcnn_mask_path)
        refined_mask = refine_mask(vibe_mask, mrcnn_mask_path)
        dst_path = os.path.join(dst_folder_path, '{}.jpg'.format(str(frame_index).zfill(5)))
        cv2.imwrite(dst_path, refined_mask)



if __name__ == '__main__':
    # VIRAT_S_040003_04_000758_001118
    # VIRAT_S_000207_02_000498_000530_F_00000227.json
    # video_folder_name = 'VIRAT_S_000207_02_000498_000530'
    # refine_main(video_folder_name)
    #
    gt_root = '../../../datasets/virat/gt_annotations/'
    videos = os.listdir(gt_root)
    # videos = ['VIRAT_S_000204_04_000738_000977', 'VIRAT_S_000205_02_000409_000566', 'VIRAT_S_000207_02_000498_000530',
    #           'VIRAT_S_040003_04_000758_001118', 'VIRAT_S_040103_00_000000_000120', 'VIRAT_S_040104_05_000939_001116']
    # videos = ['VIRAT_S_040103_00_000000_000120', 'VIRAT_S_000204_04_000738_000977', 'VIRAT_S_040003_04_000758_001118']
    videos = ['VIRAT_S_040003_04_000758_001118']

    for index,video_name in enumerate(videos):
        print 'Process {}th of {} videos'.format(index, len(videos))
        refine_main(video_name)