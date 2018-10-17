# -*- coding: utf-8 -*-
from scipy.interpolate import interp1d
import pandas
import argparse
import numpy
import json
import cPickle as pickle
import os

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

# def getDatasetDict():
#     df=pandas.read_csv("./data/activitynet_annotations/video_info_new.csv")
#     json_data= load_json("./data/activitynet_annotations/anet_anno_action.json")
#     database=json_data
#     video_dict = {}
#     for i in range(len(df)):
#         video_name=df.video.values[i]
#         video_info=database[video_name]
#         video_new_info={}
#         video_new_info['duration_frame']=video_info['duration_frame']
#         video_new_info['duration_second']=video_info['duration_second']
#         video_new_info["feature_frame"]=video_info['feature_frame']
#         video_new_info['annotations']=video_info['annotations']
#         video_new_info['subset'] = df.subset.values[i]
#         video_dict[video_name]=video_new_info
#     return video_dict

def getDatasetDict(gt_path, split_path):
    with open(gt_path, 'rb') as input_file:
        database = pickle.load(input_file)
    with open(split_path, 'rb') as input_file:
        db_splits = pickle.load(input_file)

    video_dict = {}
    for snippet_name in database:
        video_info = database[snippet_name]
        video_new_info = {}
        video_name = snippet_name.split('-')[0]
        if video_name in db_splits['train']:
            video_new_info['subset'] = 'train'
        elif video_name in db_splits['val']:
            video_new_info['subset'] = 'val'
        elif video_name in db_splits['ts']:
            video_new_info['subset'] = 'test'
        video_new_info['annotations'] = video_info['annotations']
        video_new_info['frame_inds'] = video_info['frame_inds']
        video_dict[snippet_name] = video_new_info
    return video_dict

def generateFeature(video_name, video_dict, experiment_type):

    num_sample_start=8
    num_sample_end=8
    num_sample_action=16
    num_sample_interpld = 3

    src_path = os.path.join('../../output', experiment_type, 'TEM_results/{}.csv'.format(video_name))
    # adf=pandas.read_csv("../../output/TEM_results/"+video_name+".csv")
    adf=pandas.read_csv(src_path)
    score_action=adf.action.values[:]
    seg_xmins = adf.xmin.values[:]
    seg_xmaxs = adf.xmax.values[:]
    video_scale = len(adf)
    video_gap = seg_xmaxs[0] - seg_xmins[0]
    video_extend = video_scale / 4 + 10
    src_path = os.path.join('../../output', experiment_type, 'PGM_proposals/{}.csv'.format(video_name))
    # pdf=pandas.read_csv("../../output/PGM_proposals/"+video_name+".csv")
    pdf=pandas.read_csv(src_path)

    video_subset = video_dict[video_name]['subset']
    if video_subset == "train":
        pdf=pdf[:500]
    else:
        pdf=pdf[:1000]
    tmp_zeros=numpy.zeros([video_extend])    
    score_action=numpy.concatenate((tmp_zeros,score_action,tmp_zeros))
    tmp_cell = video_gap
    tmp_x = [-tmp_cell/2-(video_extend-1-ii)*tmp_cell for ii in range(video_extend)] + \
             [tmp_cell/2+ii*tmp_cell for ii in range(video_scale)] + \
              [tmp_cell/2+seg_xmaxs[-1] +ii*tmp_cell for ii in range(video_extend)]
    f_action=interp1d(tmp_x,score_action,axis=0)
    feature_bsp=[]

    for idx in range(len(pdf)):
        xmin=pdf.xmin.values[idx]
        xmax=pdf.xmax.values[idx]
        xlen=xmax-xmin
        xmin_0=xmin-xlen/5
        xmin_1=xmin+xlen/5
        xmax_0=xmax-xlen/5
        xmax_1=xmax+xlen/5
        #start
        plen_start= (xmin_1-xmin_0)/(num_sample_start-1)
        plen_sample = plen_start / num_sample_interpld
        tmp_x_new = [ xmin_0 - plen_start/2 + plen_sample * ii for ii in range(num_sample_start*num_sample_interpld +1 )] 
        tmp_y_new_start_action=f_action(tmp_x_new)
        tmp_y_new_start = [numpy.mean(tmp_y_new_start_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_start) ]
        #end
        plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
        plen_sample = plen_end / num_sample_interpld
        tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * ii for ii in range(num_sample_end*num_sample_interpld +1 )] 
        tmp_y_new_end_action=f_action(tmp_x_new)
        tmp_y_new_end = [numpy.mean(tmp_y_new_end_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_end) ]
        #action
        plen_action= (xmax-xmin)/(num_sample_action-1)
        plen_sample = plen_action / num_sample_interpld
        tmp_x_new = [ xmin - plen_action/2 + plen_sample * ii for ii in range(num_sample_action*num_sample_interpld +1 )] 
        tmp_y_new_action=f_action(tmp_x_new)
        tmp_y_new_action = [numpy.mean(tmp_y_new_action[ii*num_sample_interpld:(ii+1)*num_sample_interpld+1]) for ii in range(num_sample_action) ]
        tmp_feature = numpy.concatenate([tmp_y_new_action,tmp_y_new_start,tmp_y_new_end])
        feature_bsp.append(tmp_feature)
    feature_bsp = numpy.array(feature_bsp)
    dst_path = os.path.join('../../output', experiment_type, 'PGM_feature/{}'.format(video_name))
    # numpy.save("../../output/PGM_feature/"+video_name,feature_bsp)
    numpy.save(dst_path,feature_bsp)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Boundary Sensitive Network")
    parser.add_argument('--experiment', default=None, help='Which folder to store samples and models')
    # parser.add_argument('start_idx', type=int)
    # parser.add_argument('end_idx', type=int)
    args = parser.parse_args()

if __name__ == '__main__':
    opt = parse_arguments()

    gt_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    split_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'

    video_dict=getDatasetDict(gt_path, split_path)
    # video_list=video_dict.keys()[args.start_idx:args.end_idx]
    video_list=video_dict.keys()

    for idx, video_name in enumerate(video_list):
        print 'Process {}th video: {}'.format(idx, video_name)
        generateFeature(video_name,video_dict, opt.experiment)
        #break