import os
import json


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

if __name__ == '__main__':
    # groundtruth of VIRAT_S_000204_04_000738_000977
    # start_frame, event_begin, event_end, end_frame

    gt_root = '../../../datasets/virat/gt_annotations/'
    folders = os.listdir(gt_root)
    event_enum = {}
    event_count = {}

    for video_folder in folders:
        gt_path = os.path.join(gt_root, video_folder, 'actv_id_type.json')
        data = load_json(gt_path)
        for event_id in data.keys():
            event_info = data[event_id]
            event_type = event_info['event_type']
            event_duration = event_info['event_end'] - event_info['event_begin'] + 1
            if event_type in event_enum:
                event_enum[event_type] += 1
            else:
                event_enum[event_type] = 0


            if event_type in event_count:
                event_count[event_type] += event_duration
            else:
                event_count[event_type] = event_duration


    event_frames = {}
    for event_type in event_enum:
        event_frames[event_type] = 1.0 * event_count[event_type]/event_enum[event_type]

    print event_frames


    # print event_enum
            # print event_info['event_type']

        # if not os.path.exists(gt_path):
        #     print gt_path




    # file_path = './actv_id_type.json'
    # data = load_json(file_path)
    # print data.keys()
    # print len(data.keys())
    #
    # for key in data.keys():
    #     event_info = data[key]





        # if event_info['event_begin'] < event_info['start_frame']:
        #     print event_info['event_begin'],event_info['event_end'],event_info['start_frame'],event_info['end_frame'],event_info['event_type']

        # if event_info['event_end'] > event_info['end_frame']:
        #     print event_info['event_begin'], event_info['event_end'], event_info['start_frame'], event_info[
        #         'end_frame'], event_info['event_type']
        # total_counter += 1
        # if event_info['event_begin'] < event_info['start_frame'] or event_info['end_frame'] < event_info['event_end']:
        #     print event_info['event_begin'],event_info['event_end'],event_info['start_frame'],event_info['end_frame'],event_info['event_type']
        #     counter += 1
        # assert False
    # print total_counter, counter

    # print data.keys()