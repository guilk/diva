import os
import numpy as np
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def compute_overlap(event_list):
    event_list = sorted(event_list, key=lambda x: x[0])
    print event_list
    counter = 0
    for cur_index, event_duration in enumerate(event_list):
        cur_start_time = event_duration[0]
        cur_end_time = event_duration[1]
        for next_index in range(cur_index+1, len(event_list)):
            next_start_time = event_list[next_index][0]
            next_end_time = event_list[next_index][1]
            if next_start_time > cur_start_time and next_start_time < cur_end_time and next_end_time < cur_end_time:
                counter += 2
    return counter

def plot_stats(event_lens):

    nbins = len(event_lens)
    # fig, ax = plt.subplots()

    plt.hist(event_lens, bins=100)
    plt.tight_layout()
    plt.savefig('./num_events.jpg')

def get_events(gt_root, event_set):
    folders = os.listdir(gt_root)
    event_stats = {}
    count = 0
    event_lens = []


    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        duration_list = []
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            if event_type in event_set:
                duration_list.append((event_info['event_begin'], event_info['event_end']))
                event_lens.append(event_info['event_end']-event_info['event_begin']+1)
        if not duration_list:
            continue
        # print duration_list

        count += compute_overlap(duration_list)

    # plot_stats(event_lens)
    # print max(event_lens), min(event_lens), 1.0*sum(event_lens)/len(event_lens)
            #
            # event_duration = event_info['event_end'] - event_info['event_begin'] + 1
            # if event_type not in event_stats:
            #     event_stats[event_type] = [event_duration]
            # else:
            #     event_stats[event_type].append(event_duration)

    print 'The number of overlapped videos is: {} out of {} exmaples'.format(count, len(event_lens))
    return event_stats


def get_event_types(gt_root):
    folders = os.listdir(gt_root)
    event_types = set()

    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            event_types.add(event_type)
    return event_types


if __name__ == '__main__':
    gt_root = '../../../datasets/virat/gt_annotations/'
    # event_types = get_event_types(gt_root)
    event_types = set(['Loading', 'Unloading', 'Open_Trunk', 'Closing_Trunk', 'Opening', 'Closing', 'Exiting', 'Entering'])
    get_events(gt_root, event_types)
    # print event_types