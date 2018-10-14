import random
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


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


def generate_bbox_events(gt_path):
    data = load_json(gt_path)
    bbox_infos = {}

    for event_id in data.keys():
        event_info = data[event_id]
        # event_info = data['50001']
        event_type = event_info['event_type']
        event_color = generate_colors()[0]
        event_objects = event_info['objects']
        # print event_objects
        for object_id in event_objects:
            # print object_id
            object_info = event_objects[object_id]
            bbox_dict = object_info['trajectory']
            for frame_id in bbox_dict:
                # bbox_info = []
                # bbox_info.append([event_color, bbox_dict[frame_id], event_type])
                bbox_info = (event_color, bbox_dict[frame_id], event_type)

                if int(frame_id) not in bbox_infos:
                    bbox_infos[int(frame_id)] = [bbox_info]
                else:
                    bbox_infos[int(frame_id)].append(bbox_info)
    return bbox_infos

def print_events(gt_path):
    data = load_json(gt_path)

    for event_id in data.keys():
        event_info = data[event_id]
        print event_info['event_begin'], event_info['event_end'], event_info['event_type']



def draw_visulizations(video_folder):
    gt_root = '../../../datasets/virat/gt_annotations/'
    img_root = os.path.join('../../../datasets/virat/frames/', video_folder)
    # img_root = '../../../datasets/virat/resized_frames/VIRAT_S_000204_04_000738_000977'
    dst_root = os.path.join('../../../datasets/virat/visualized_frames/', video_folder)
    # dst_root = '../../../datasets/virat/visualized_frames/VIRAT_S_000204_04_000738_000977'
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # video_folder = 'VIRAT_S_000204_04_000738_000977'
    gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
    # print_events(gt_path)

    bbox_infos = generate_bbox_events(gt_path)

    imgs = os.listdir(img_root)
    imgs = sorted(imgs)
    for img_name in imgs:
        # print 'Process {}th image'.format(img_name)
        img = cv2.imread(os.path.join(img_root, img_name))
        img_ind = int(img_name.split('.')[0])
        img_ind += 1

        if img_ind in bbox_infos:
            # print img_ind
            # print bbox_infos[img_ind]
            # assert False
            for bbox_info in bbox_infos[img_ind]:
                event_color = bbox_info[0]
                bbox = bbox_info[1]
                event_type = bbox_info[2]

                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), event_color, 1)
                cv2.putText(img, event_type, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, event_color, lineType = cv2.LINE_AA)
        cv2.imwrite(os.path.join(dst_root, img_name), img)


if __name__ == '__main__':

    video_root = '../../../datasets/virat/gt_annotations/'
    videos = os.listdir(video_root)

    # videos = ['VIRAT_S_000207_02_000498_000530']
    for index, video_folder in enumerate(videos):
        print 'Process {}th video of {}'.format(index, len(videos))
        # video_folder = video_name.split('.')[0]
        draw_visulizations(video_folder)

    # # groundtruth of VIRAT_S_000204_04_000738_000977
    # # 1280 * 720
    # # start_frame, event_begin, event_end, end_frame
    #
    # gt_root = '../../../datasets/virat/gt_annotations/'
    # img_root = '../../../datasets/virat/resized_frames/VIRAT_S_000204_04_000738_000977'
    # dst_root = '../../../datasets/virat/visualized_frames/VIRAT_S_000204_04_000738_000977'
    # if not os.path.exists(dst_root):
    #     os.makedirs(dst_root)
    #
    # video_folder = 'VIRAT_S_000204_04_000738_000977'
    # gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
    # print_events(gt_path)
    #
    # bbox_infos = generate_bbox_events(gt_path)
    #
    # imgs = os.listdir(img_root)
    # imgs = sorted(imgs)
    # for img_name in imgs:
    #     # print 'Process {}th image'.format(img_name)
    #     img = cv2.imread(os.path.join(img_root, img_name))
    #     img_ind = int(img_name.split('.')[0])
    #     img_ind += 1
    #
    #     if img_ind in bbox_infos:
    #         # print img_ind
    #         # print bbox_infos[img_ind]
    #         # assert False
    #         for bbox_info in bbox_infos[img_ind]:
    #             event_color = bbox_info[0]
    #             bbox = bbox_info[1]
    #             event_type = bbox_info[2]
    #
    #             cv2.rectangle(img, (bbox[0]/2, bbox[1]/2), (bbox[2]/2, bbox[3]/2), event_color, 1)
    #             cv2.putText(img, event_type, (bbox[0]/2, bbox[1]/2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, event_color, lineType = cv2.LINE_AA)
    #     cv2.imwrite(os.path.join(dst_root, img_name), img)