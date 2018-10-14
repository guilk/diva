import os
import numpy as np
import json
import cPickle as pickle
from random import shuffle
import math

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def get_uniform_snippets(num_frames):
    len_snippet = 300
    frame_inds = range(num_frames)
    start_frames = frame_inds[::len_snippet]
    uniform_intervals = []
    for start_frame in start_frames:
        end_frame = start_frame + len_snippet - 1
        if start_frame + len_snippet >= num_frames:
            end_frame = num_frames - 1
        uniform_intervals.append((start_frame, end_frame))
    return uniform_intervals

def get_gt_snippets(gt_path, event_set):
    '''
    :param gt_path: the ground-truth file
    :param event_set: selected event types
    :return: ground-truth snippets given event types
    '''
    data = load_json(gt_path)
    gt_snippets = []
    for event_id in data.keys():
        event_info = data[event_id]
        event_type = event_info['event_type']
        if event_type in event_set:
            gt_snippets.append((event_info['event_begin'], event_info['event_end'], event_info['event_type']))
    gt_snippets = sorted(gt_snippets, key=lambda x:x[0])
    return gt_snippets

def get_sliding_windows(num_frames):
    len_snippet = 300
    len_stride = 100
    frame_inds = range(num_frames)
    start_frames = frame_inds[::len_stride]
    sliding_snippets = []
    for start_frame in start_frames:
        end_frame = start_frame + len_snippet - 1
        if start_frame + len_snippet >= num_frames:
            end_frame = num_frames - 1
        sliding_snippets.append((start_frame, end_frame))
    return sliding_snippets


def compute_overlap(candidate_snippet, gt_snippets):

    match_snippets = []

    candidate_start_frame = candidate_snippet[0]
    candidate_end_frame = candidate_snippet[1]

    for gt_snippet in gt_snippets:
        gt_start_frame = gt_snippet[0]
        gt_end_frame = gt_snippet[1]
        if gt_start_frame >= candidate_start_frame and gt_start_frame <= candidate_end_frame:
            match_snippets.append(gt_snippet)
        elif gt_end_frame >= candidate_start_frame and gt_end_frame <= candidate_end_frame:
            match_snippets.append(gt_snippet)
    return match_snippets

def generate_splits(gt_root):
    video_folders = os.listdir(gt_root)
    shuffle(video_folders)
    num_videos = len(video_folders)
    num_train_videos = int(math.ceil(0.7 * num_videos))
    num_val_videos = int(math.ceil(0.1 * num_videos))
    num_ts_videos = int(math.ceil(0.2 * num_videos))
    train_list = video_folders[:num_train_videos]
    val_list = video_folders[num_train_videos:num_train_videos+num_val_videos]
    ts_list = video_folders[num_train_videos+num_val_videos:]

    split_dict = {}
    split_dict['train'] = train_list
    split_dict['val'] = val_list
    split_dict['ts'] = ts_list
    return split_dict

def generate_groundtruth(gt_path, split_path):
    # gt_path = '../../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    # split_path = '../../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'
    with open(gt_path, 'rb') as input_file:
        database = pickle.load(input_file)
    with open(split_path, 'rb') as input_file:
        db_splits = pickle.load(input_file)

    data = {}
    version = 'VIRAT-19'
    taxonomy = {}

    for snippet_name in database:
        snippet_info = database[snippet_name]
        snippet_new_info = {}
        video_name = snippet_name.split('-')[0]
        if video_name in db_splits['train']:
            snippet_new_info['subset'] = 'train'
        elif video_name in db_splits['val']:
            snippet_new_info['subset'] = 'validation'
        elif video_name in db_splits['ts']:
            snippet_new_info['subset'] = 'test'
        # annotations = []
        frame_inds = snippet_info['frame_inds']
        segments = []
        for ann in snippet_info['annotations']:
            segment_info = {}
            actual_start_frame = ann[0] - frame_inds[0]
            actual_end_frame = ann[1] - frame_inds[0]
            snippet_start_frame = 0
            snippet_end_frame = frame_inds[1] - frame_inds[0]
            gt_start_frame = max(snippet_start_frame, actual_start_frame)
            gt_end_frame = min(snippet_end_frame, actual_end_frame)
            label = ann[2]
            segment_info['label'] = label
            segment_info['segment'] = (gt_start_frame, gt_end_frame)
            segments.append(segment_info)
            # print gt_start_frame, gt_end_frame, label
        snippet_new_info['annotations'] = segments
        data[snippet_name] = snippet_new_info

    print data['VIRAT_S_040101_03_000460_000551-700']
    output_dict = {"version": version, "database": data, "taxonomy": taxonomy}
    outfile=open("../Evaluation/data/virat_stride100_interval_300.json","w")
    json.dump(output_dict,outfile)
    outfile.close()


if __name__ == '__main__':
    # '''
    # data format: video_name-snippet_start(dict): {frame_ids(candidate_snippets), annotations(matched_gt_truth)}
    #
    # set([u'vehicle_u_turn', u'Pull', u'Loading', u'Open_Trunk', u'Closing_Trunk', u'activity_carrying', u'Opening',
    # u'Exiting', u'Talking', u'specialized_talking_phone', u'Transport_HeavyCarry', u'Entering',
    # u'specialized_texting_phone', u'vehicle_turning_right', u'Riding', u'vehicle_turning_left', u'Interacts',
    # u'Closing', u'Unloading'])
    # '''
    #
    # gt_root = '../../../datasets/virat/gt_annotations/'
    # frames_root = '../../../datasets/virat/resized_frames'
    # event_set = set(['Loading', 'Unloading', 'Open_Trunk', 'Closing_Trunk', 'Opening', 'Closing', 'Exiting', 'Entering'])
    #
    # dst_root = '../../../datasets/virat/bsn_dataset/stride_100_interval_300'
    # if not os.path.exists(dst_root):
    #     os.makedirs(dst_root)
    #
    # total_count = 0
    # folders = os.listdir(gt_root)
    # video_dict = {}
    #
    # for video_folder in folders:
    #     frames_path = os.path.join(frames_root, video_folder)
    #     frames = [frame_name for frame_name in os.listdir(frames_path)
    #               if os.path.exists(os.path.join(frames_path, frame_name))]
    #     num_frames = len(frames)
    #     # candidate_snippets = get_uniform_snippets(num_frames)
    #     candidate_snippets = get_sliding_windows(num_frames)
    #     # print len(candidate_snippets)
    #
    #     gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
    #     gt_snippets = get_gt_snippets(gt_path, event_set)
    #
    #     for candidate_snippet in candidate_snippets:
    #         snippet_dict = {}
    #         matched_snippets = compute_overlap(candidate_snippet, gt_snippets)
    #         if matched_snippets:
    #             snippet_dict['frame_inds'] = candidate_snippet
    #             snippet_dict['annotations'] = matched_snippets
    #             video_dict['{}-{}'.format(video_folder, candidate_snippet[0])] = snippet_dict
    #             # print candidate_snippet,matched_snippets
    #
    # print 'Total number of samples is {}'.format(len(video_dict))
    # with open(os.path.join(dst_root, 'gt_annotations.pkl'), 'wb') as fp:
    #     pickle.dump(video_dict, fp)
    #
    # split_dict = generate_splits(gt_root)
    # with open(os.path.join(dst_root, 'split.pkl'), 'wb') as fp:
    #     pickle.dump(split_dict, fp)

    gt_path = '../../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    split_path = '../../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'

    generate_groundtruth(gt_path, split_path)
