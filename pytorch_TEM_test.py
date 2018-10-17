import os
import cPickle
import torch.nn as nn
import pandas as pd


import torch
from TEM_model import TEM
import argparse
import torch.optim as optim
import numpy as np
import TEM_load_data
import pytorch_TEM_load_data as TEM_load_data
import cPickle as pickle

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
    parser.add_argument('--experiment', default=None, help='which folder to store samples and models')
    parser.add_argument('--embedsize', type=int, default=64, help='embedding size of input feature')
    parser.add_argument('--hiddensize', type=int, default=128, help='hidden size of network')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_arguments()
    batch_size = opt.batchSize
    if opt.experiment == None:
        opt.experiment = './pytorch_models'
        output_root = '../../output'
    else:
        output_root = os.path.join('../../output/', opt.experiment)
        opt.experiment = os.path.join('./pytorch_models', opt.experiment)

    # video_dict = TEM_load_data.load_json("./data/activitynet_annotations/anet_anno_action.json")
    gt_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    with open(gt_path, 'rb') as input_file:
        video_dict = pickle.load(input_file)
    feat_dict = TEM_load_data.load_whole_features()

    batch_result_action = []
    batch_result_start = []
    batch_result_end = []
    batch_result_xmin = []
    batch_result_xmax = []

    batch_video_list = TEM_load_data.getBatchListTest(video_dict, batch_size, shuffle=False)

    tem = TEM()
    model_path = os.path.join(opt.experiment, 'TEM/tem_model_best.pth')
    tem.load_state_dict(torch.load(model_path))
    tem.cuda()

    for idx in range(len(batch_video_list)):
        print 'Process {}th of {} batch'.format(idx, len(batch_video_list))
        batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature=\
            TEM_load_data.getProposalDataTest(batch_video_list[idx],feat_dict)
        batch_anchor_feature = np.transpose(batch_anchor_feature, (0, 2, 1))
        X_feature = torch.FloatTensor(batch_anchor_feature).cuda()
        anchors = tem(X_feature)
        anchors_action = anchors[:, 0, :]
        anchors_start = anchors[:, 1, :]
        anchors_end = anchors[:, 2, :]

        batch_result_action.append(anchors_action.data.cpu().numpy())
        batch_result_start.append(anchors_start.data.cpu().numpy())
        batch_result_end.append(anchors_end.data.cpu().numpy())
        batch_result_xmin.append(batch_anchor_xmin)
        batch_result_xmax.append(batch_anchor_xmax)


    columns=["action","start","end","xmin","xmax"]
    # output_root = '../../output/lr_{}_niter_{}_batchsize_{}_embedsize_{}_hiddensize_{}_stepsize_{}_gamma_{}'\
    #     .format(opt.lr, opt.niter, opt.batchsize, opt.embedsize, opt.hiddensize, opt.stepsize, opt.gamma)

    for idx in range(len(batch_video_list)):
        b_video=batch_video_list[idx]
        b_action=batch_result_action[idx]
        b_start=batch_result_start[idx]
        b_end=batch_result_end[idx]
        b_xmin=batch_result_xmin[idx]
        b_xmax=batch_result_xmax[idx]
        for j in range(len(b_video)):
            tmp_video=b_video[j]
            tmp_action=b_action[j]
            tmp_start=b_start[j]
            tmp_end=b_end[j]
            tmp_xmin=b_xmin[j]
            tmp_xmax=b_xmax[j]
            tmp_result=np.stack((tmp_action,tmp_start,tmp_end,tmp_xmin,tmp_xmax),axis=1)
            tmp_df=pd.DataFrame(tmp_result,columns=columns)
            # tmp_df.to_csv("../../output/TEM_results/"+tmp_video+".csv",index=False)
            tmp_df.to_csv(os.path.join(output_root, 'TEM_results/{}.csv'.format(tmp_video)), index=False)