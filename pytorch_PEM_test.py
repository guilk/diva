import os
import cPickle
import torch.nn as nn


import torch
from PEM_model import PEM
import argparse
import torch.optim as optim
# import PEM_load_data
import pytorch_PEM_load_data as PEM_load_data
import numpy as np
import pandas as pd


def abs_smooth(x):
    '''
    Smoothed absolute function. Useful to compute an L1 smooth error.
    :param x:
    :return:
    '''
    absx = torch.abs(x)
    minx = torch.min(absx, torch.ones_like(absx))
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def PEM_loss(anchors_iou, match_iou):
    u_ratio_m = 1
    u_ratio_l = 2

    # iou regressor
    u_hmask = (match_iou > 0.6).type(torch.cuda.FloatTensor)
    u_mmask = ((match_iou >= 0.2) & (match_iou <= 0.6)).type(torch.cuda.FloatTensor)
    u_lmask = (match_iou < 0.2).type(torch.cuda.FloatTensor)

    num_h = u_hmask.sum()
    num_m = u_mmask.sum()
    num_l = u_lmask.sum()

    r_m = u_ratio_m * num_h / num_m
    r_m = torch.min(r_m, torch.ones_like(r_m))
    u_smmask = torch.rand(u_hmask.size()[0]).cuda()
    # u_smmask = torch.FloatTensor(u_smmask).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1.0 - r_m)).type(torch.cuda.FloatTensor)

    r_l = u_ratio_l * num_h/num_l
    r_l = torch.min(r_l, torch.ones_like(r_l))
    u_slmask = torch.rand(u_hmask.size()[0]).cuda()
    # u_slmask = torch.FloatTensor(u_slmask).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1.0 - r_l)).type(torch.cuda.FloatTensor)

    iou_weights = u_hmask + u_smmask + u_slmask
    iou_loss = abs_smooth(match_iou - anchors_iou)
    num_nonzero = iou_weights.sum()
    weighted_loss = iou_loss * iou_weights
    iou_loss = weighted_loss.sum() / num_nonzero
    return iou_loss


def run_pem(pem, X, Y_iou):
    net_output = pem(X)
    anchors_iou = net_output.view(-1)
    loss = 10 * PEM_loss(anchors_iou, Y_iou)
    return loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_arguments()
    batch_size = opt.batchSize
    num_epoches = opt.niter
    if opt.experiment == None:
        experiment_type = ''
        opt.experiment = './pytorch_models'
    else:
        experiment_type = opt.experiment
        opt.experiment = os.path.join('./pytorch_models', opt.experiment)


    pem = PEM()
    model_path = os.path.join(opt.experiment, 'PEM/pem_model_best.pth')
    pem.load_state_dict(torch.load(model_path))
    pem.cuda()

    gt_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    split_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'
    train_dict, val_dict, test_dict = PEM_load_data.getDatasetDict(gt_path, split_path)
    FullDict = PEM_load_data.getTestData(train_dict, val_dict, test_dict, "validation", experiment_type)
    batch_video_list = PEM_load_data.getBatchList(val_dict, batch_size)
    video_list = val_dict.keys()
    for idx in range(len(video_list)):
        video_name = video_list[idx]
        prop_dict = FullDict[video_name]
        batch_feature, batch_iou_list, batch_ioa_list = \
            PEM_load_data.prop_dict_data({"data": prop_dict})
        X_feature = torch.FloatTensor(batch_feature).cuda()
        out_score = pem(X_feature)
        anchors_iou = out_score.view(-1)
        iou_score = anchors_iou.data.cpu().numpy()

        xmin_list = prop_dict["xmin"]
        xmax_list = prop_dict["xmax"]
        xmin_score_list = prop_dict["xmin_score"]
        xmax_score_list = prop_dict["xmax_score"]
        latentDf = pd.DataFrame()
        latentDf["xmin"] = xmin_list
        latentDf["xmax"] = xmax_list
        latentDf["xmin_score"] = xmin_score_list
        latentDf["xmax_score"] = xmax_score_list
        latentDf["iou_score"] = iou_score

        # latentDf.to_csv("../../output/PEM_results/" + video_name + ".csv", index=False)
        latentDf.to_csv(os.path.join('../../output', experiment_type, 'PEM_results/{}.csv'.format(video_name)),
                        index=False)