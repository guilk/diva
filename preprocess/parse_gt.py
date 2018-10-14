import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_events(gt_root):
    folders = os.listdir(gt_root)
    event_stats = {}

    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            event_duration = event_info['event_end'] - event_info['event_begin'] + 1
            if event_type not in event_stats:
                event_stats[event_type] = [event_duration]
            else:
                event_stats[event_type].append(event_duration)

    return event_stats

def get_event_infos(gt_root):
    folders = os.listdir(gt_root)
    event_stats = {}
    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        event_list = []
        # event_name = ''
        for event_id in data.keys():
            event_info = data[event_id]
            event_list.append((event_info['event_begin'], event_info['event_end'], event_info['event_type']))
            # event_name = event_info['event_type']

        # calculate the stats of events
        calculate_overlap(event_list)


    return event_stats

def calculate_overlap(event_list):
    sorted_list = sorted(event_list, key=lambda x: x[0])

def get_mean_std(event_durations):
    event_names = []
    event_means = []
    event_stds = []
    event_nums = []

    for event_name in event_durations:
        event_names.append(event_name)
        event_means.append(np.mean(np.asarray(event_durations[event_name])))
        event_stds.append(np.std(np.asarray(event_durations[event_name])))
        event_nums.append(len(event_durations[event_name]))

    return event_names, event_means, event_stds, event_nums


def plot_mean_std(event_names, event_means, event_stds):
    x_pos = np.arange(len(event_names))
    fig, ax = plt.subplots()
    ax.bar(x_pos, event_means, yerr = event_stds, align='center', alpha = 0.5, ecolor = 'black', capsize = 10)
    ax.set_ylabel('Durations of events (frames)')

    xtickNames = plt.setp(ax, xticklabels=event_names)
    plt.setp(xtickNames, rotation=60, fontsize=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(event_names)
    ax.set_title('Stats of the DIVA dataset')
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('./stats_diva.png')

def plot_event_nums(event_names, event_nums):
    for event_name, event_num in zip(event_names, event_nums):
        print event_name, event_num
    assert False
    x_pos = np.arange(len(event_names))
    fig, ax = plt.subplots()
    ax.bar(x_pos, event_nums)
    ax.set_ylabel('Number of events')
    xtickNames = plt.setp(ax, xticklabels=event_names)
    plt.setp(xtickNames, rotation=60, fontsize=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(event_names)
    ax.set_title('Stats of the DIVA dataset (Total: 2599)')
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('./nums_diva.png')

def test(gt_root):
    folders = os.listdir(gt_root)
    event_stats = {}

    for video_folder in folders:
        if video_folder != 'VIRAT_S_000204_04_000738_000977':
            continue
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        event_list = []
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            event_begin = event_info['event_begin']
            event_end = event_info['event_end']
            event_list.append((event_type, event_begin, event_end))
        print event_list
    return event_stats


def get_events_set(gt_root):
    folders = os.listdir(gt_root)
    events_set = set()

    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            events_set.add(event_type)
    print events_set


def visulize_events(gt_path):
    data = load_json(gt_path)

    for event_id in data.keys():
        event_info = data[event_id]
        event_type = event_info['event_type']

        #
        #
        # print event_info['event_type']
        # print event_info['objects']



if __name__ == '__main__':
    # groundtruth of VIRAT_S_000204_04_000738_000977
    # 1280 * 720
    # start_frame, event_begin, event_end, end_frame

    gt_root = '../../../datasets/virat/gt_annotations/'
    get_events_set(gt_root)
    # get_event_infos(gt_root)
    #
    # event_durations = get_events(gt_root)
    #
    # event_names, event_means, event_stds, event_nums = get_mean_std(event_durations)

    # plot_mean_std(event_names, event_means, event_stds)
    #
    # plot_event_nums(event_names, event_nums)

