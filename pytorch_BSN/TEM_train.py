import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from TEM_model import TEM
import argparse


import os
import cPickle


def binary_logistic_loss(gt_scores, pred_anchors):
    '''
    Calculate weighted binary logistic loss
    :param gt_scores:
    :param pred_anchors:
    :return:
    '''
    gt_scores = gt_scores.view(-1)
    pred_anchors = pred_anchors.view(-1)
    pmask = (gt_scores>0.5).type(torch.cuda.FloatTensor)
    num_positive = pmask.sum()
    num_entries = gt_scores.size()[0]
    ratio = num_entries/num_positive

    coef_0 = 0.5 * ratio / (ratio-1)
    coef_1 = coef_0 * (ratio-1)
    loss = coef_1*pmask*torch.log(pred_anchors) + coef_0*(1.0-pmask)*torch.log(1.0-pred_anchors)
    loss = -torch.mean(loss)
    num_sample = [num_positive, ratio]
    return loss, num_sample


def train(tem_model, X_feature, Y_action, Y_start, Y_end):
    anchors = tem_model(X_feature)

    anchors_action = anchors[:,:,0]
    anchors_start = anchors[:,:,1]
    anchors_end = anchors[:,:,2]

    loss_action = binary_logistic_loss(Y_action, anchors_action)
    loss_start = binary_logistic_loss(Y_start, anchors_start)
    loss_end = binary_logistic_loss(Y_end, anchors_end)
    loss = 2 * loss_action + loss_start + loss_end

    return loss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_arguments()

    # Intialize model
    tem = TEM()
    if opt.cuda:
        tem.cuda()


















    # data_root = '../../../loss_test/tem'
    # loss_files = os.listdir(data_root)
    #
    # for loss_file in loss_files:
    #     file_path = os.path.join(data_root, loss_file)
    #     with open(file_path, 'rb') as input_file:
    #         loss_dict = cPickle.load(input_file)
    #
    #     anchors_action = loss_dict['anchors_action']
    #     Y_action = loss_dict['Y_action']
    #     loss_action = loss_dict['loss_action']
    #     num_sample_action = loss_dict['num_sample_action']
    #
    #     anchors_action = torch.FloatTensor(anchors_action).cuda()
    #     Y_action = torch.FloatTensor(Y_action).cuda()
    #     # anchors_action = torch.from_numpy(anchors_action)
    #     # Y_action = torch.from_numpy(Y_action)
    #
    #     loss, num_samples = binary_logistic_loss(Y_action, anchors_action)
    #     # print type(loss)
    #     # print loss.is_cuda
    #     cpu_loss = loss.cpu().numpy()
    #     print cpu_loss, loss_action
    #     assert (cpu_loss - loss_action) < 0.0001
    #     # print loss.cpu().numpy(),loss_action
    #
    #     # print type(anchors_action)
    #     # print type(Y_action)
    #     # print anchors_action.is_cuda
    #     # print Y_action.is_cuda
    #     # assert False

