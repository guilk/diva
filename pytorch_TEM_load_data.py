import random
import numpy as np
import pandas as pd
import json
import cPickle as pickle
import os

tscale = 100
tgap = 1. / tscale
len_window = 300


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def load_features(feat_root, feat_type, video_name):
    feat_path = os.path.join(feat_root, 'presaved_{}'.format(feat_type), '{}.npy'.format(video_name))
    snippet_feature = np.load(feat_path)
    return snippet_feature


def load_save_features(feat_root, feat_type, video_name, frame_inds):
    frame_inds = range(frame_inds[0], frame_inds[1]+1)
    if len(frame_inds) < len_window:
        frame_inds = frame_inds + [frame_inds[-1]]*(len_window - len(frame_inds))
    frame_interval = int(len_window/tscale)
    frame_inds = frame_inds[::frame_interval]

    video_folder = video_name.split('-')[0]
    video_feat_folder = os.path.join(feat_root, feat_type, video_folder)
    frame_files = os.listdir(video_feat_folder)
    frame_files = sorted(frame_files)
    padding_frame_file = frame_files[-1]

    frame_features = []
    for frame_index in frame_inds:
        frame_path = os.path.join(video_feat_folder, '{}.npy'.format(str(frame_index).zfill(5)))
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_feat_folder, padding_frame_file)
        feature = np.load(frame_path)
        frame_features.append(feature)
    frame_features = np.asarray(frame_features)
    frame_features = np.transpose(frame_features, [0,3,1,2])

    dst_folder_path = os.path.join(feat_root, 'presaved_{}'.format(feat_type))
    if not os.path.exists(dst_folder_path):
        os.makedirs(dst_folder_path)
    dst_feat_path = os.path.join(dst_folder_path, '{}.npy'.format(video_name))
    np.save(dst_feat_path, frame_features)
    print frame_features.shape
    return frame_features

def getDatasetDict(gt_path, split_path):
    """Load dataset file
    """
    with open(gt_path, 'rb') as input_file:
        database = pickle.load(input_file)
    with open(split_path, 'rb') as input_file:
        db_splits = pickle.load(input_file)

    train_dict = {}
    val_dict = {}
    test_dict = {}

    for snippet_name in database:
        snippet_info = database[snippet_name]
        # {'annotations': [(2974, 3147, u'Unloading')], 'frame_inds': (3000, 3299)}
        video_name = snippet_name.split('-')[0]
        if video_name in db_splits['train']:
            train_dict[snippet_name] = snippet_info
        elif video_name in db_splits['val']:
            val_dict[snippet_name] = snippet_info
        elif video_name in db_splits['ts']:
            test_dict[snippet_name] = snippet_info
    return train_dict, val_dict, test_dict

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def getBatchList(numVideo, batch_size, shuffle=True):
    """Generate batch list for each epoch randomly
    """
    video_list = range(numVideo)
    batch_start_list = [i * batch_size for i in range(len(video_list) / batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    if shuffle == True:
        random.shuffle(video_list)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart:(bstart + batch_size)])
    return batch_video_list


def getBatchListTest(video_dict, batch_size, shuffle=True):
    """Generate batch list during testing
    """
    video_list = video_dict.keys()
    batch_start_list = [i * batch_size for i in range(len(video_list) / batch_size)]
    batch_start_list.append(len(video_list) - batch_size)
    if shuffle == True:
        random.shuffle(video_list)
    batch_video_list = []
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart:(bstart + batch_size)])
    return batch_video_list


def getBatchData(video_list, data_dict):
    """Given a video list (batch), get corresponding data
    """
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []
    batch_anchor_feature = []

    for idx in video_list:
        batch_label_action.append(data_dict["gt_action"][idx])
        batch_label_start.append(data_dict["gt_start"][idx])
        batch_label_end.append(data_dict["gt_end"][idx])
        batch_anchor_feature.append(data_dict["feature"][idx])

    batch_label_action = np.array(batch_label_action)
    batch_label_start = np.array(batch_label_start)
    batch_label_end = np.array(batch_label_end)
    batch_anchor_feature = np.array(batch_anchor_feature)
    batch_anchor_feature = np.reshape(batch_anchor_feature, [len(video_list), tscale, -1])
    return batch_label_action, batch_label_start, batch_label_end, batch_anchor_feature


def getFullData(train_dict, val_dict, test_dict, dataSet, features):
    """Load full data in dataset
    """
    if dataSet == "train":
        video_dict = train_dict
    elif dataSet == 'val':
        video_dict = val_dict
    elif dataSet == 'test':
        video_dict = test_dict
    video_list = video_dict.keys()

    batch_bbox = []
    batch_index = [0]
    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []

    for i in range(len(video_list)):
        if i % 100 == 0:
            print "%d / %d %s videos are loaded" % (i, len(video_list), dataSet)
        video_name = video_list[i]

        video_info = video_dict[video_name]
        frame_inds = video_info['frame_inds']
        start_frame = frame_inds[0]

        video_labels = video_info['annotations']
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = tmp_info[0]
            tmp_end = tmp_info[1]
            tmp_start = 1.0 * (tmp_start - start_frame)/len_window
            tmp_end = 1.0 * (tmp_end - start_frame)/len_window
            batch_bbox.append([tmp_start, tmp_end])

        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        batch_index.append(batch_index[-1] + len(video_labels))
        # snippet_feature = load_features(feat_root, feat_type, video_name)
        snippet_feature = features[video_name]
        batch_anchor_feature.append(snippet_feature)
        # snippet_features = load_save_features(feat_root, feat_type, video_name,frame_inds)
        # batch_anchor_feature.append(snippet_features)

    num_data = len(batch_anchor_feature)
    batch_label_action = []
    batch_label_start = []
    batch_label_end = []

    for idx in range(num_data):
        gt_bbox=np.array(batch_bbox[batch_index[idx]:batch_index[idx+1]])
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]
        anchor_xmin=batch_anchor_xmin[idx]
        anchor_xmax=batch_anchor_xmax[idx]

        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(tgap, 0.1 * gt_lens)

        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)

        match_score_action = []
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(np.max(ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(
                np.max(ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))

        batch_label_action.append(match_score_action)
        batch_label_start.append(match_score_start)
        batch_label_end.append(match_score_end)
    dataDict = {"gt_action": batch_label_action, "gt_start": batch_label_start, "gt_end": batch_label_end,
                "feature": batch_anchor_feature}
    return dataDict


def getProposalDataTest(video_list, feat_dict):
    """Load data during testing
    """
    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []
    for i in range(len(video_list)):
        video_name = video_list[i]
        tmp_anchor_xmin = [tgap * i for i in range(tscale)]
        tmp_anchor_xmax = [tgap * i for i in range(1, tscale + 1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))
        batch_anchor_xmax.append(list(tmp_anchor_xmax))
        # tmp_df=pd.read_csv("./data/activitynet_feature_cuhk/csv_mean_"+str(tscale)+"/"+video_name+".csv")
        # tmp_df = pd.read_csv("../../datasets/csv_mean_" + str(tscale) + "/" + video_name + ".csv")
        tmp_feat = feat_dict[video_name]
        batch_anchor_feature.append(tmp_feat)
    batch_anchor_xmin = np.array(batch_anchor_xmin)
    batch_anchor_xmax = np.array(batch_anchor_xmax)
    batch_anchor_feature = np.array(batch_anchor_feature)
    batch_anchor_feature = np.reshape(batch_anchor_feature, [len(video_list), tscale, -1])
    return batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature


def save_pkl_features():
    feat_root = '/home/liangke/diva/datasets/virat/features/presaved_rgb_features'
    feat_files = os.listdir(feat_root)
    feat_dict = {}
    for feat_file in feat_files:
        feat_path = os.path.join(feat_root, feat_file)
        print feat_path
        snippet_feature = np.load(feat_path)
        # Due to the memory limit, compute the average pooling of the spatial dimension
        snippet_feature = snippet_feature.mean(axis=(2, 3))
        video_name = feat_file.split('.')[0]
        feat_dict[video_name] = snippet_feature
    with open(os.path.join(feat_root, 'features.pkl'), 'wb') as fw:
        pickle.dump(feat_dict, fw)


def load_whole_features():
    feat_root = '../../datasets/virat/features/'
    feat_type = 'rgb_features'
    feat_path = os.path.join(feat_root, 'presaved_{}'.format(feat_type), 'features.pkl')
    with open(feat_path, 'rb') as input_file:
        rgb_features = pickle.load(input_file)
    return rgb_features

if __name__ == '__main__':
    gt_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    split_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'

    # save_pkl_features()
    rgb_features = load_whole_features()

    train_dict, val_dict, test_dict = getDatasetDict(gt_path, split_path)
    getFullData(train_dict, val_dict, test_dict, 'train', rgb_features)
    getFullData(train_dict, val_dict, test_dict, 'val', rgb_features)
    # getFullData(train_dict, val_dict, test_dict, 'test', rgb_features)