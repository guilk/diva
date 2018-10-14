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
        opt.experiment = './pytorch_models'

    pem = PEM()
    pem.cuda()

    optimizer = optim.Adam(pem.parameters(), lr = 0.001, betas=(opt.beta1, 0.999), weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    gt_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/gt_annotations.pkl'
    split_path = '../../datasets/virat/bsn_dataset/stride_100_interval_300/split.pkl'

    train_dict, val_dict, test_dict = PEM_load_data.getDatasetDict(gt_path, split_path)
    train_data = PEM_load_data.getTrainData(train_dict, val_dict, test_dict, batch_size, "train")
    val_data = PEM_load_data.getTrainData(train_dict, val_dict, test_dict, batch_size, "validation")

    train_info = {'iou_loss' : [], 'l2' : []}
    val_info = {'iou_loss':[], 'l2':[]}

    best_val_cost = 1000000

    for epoch in range(num_epoches):
        # train
        scheduler.step()
        for p in pem.parameters():
            p.requires_grad = True
        pem.train()
        mini_info = {'iou_loss':[], 'l2':[]}
        for idx in range(len(train_data)):
            # print 'Process {}th batch'.format(idx)
            prop_dict = train_data[idx]
            batch_feature,batch_iou_list,batch_ioa_list=PEM_load_data.prop_dict_data(prop_dict)

            X_feature = torch.FloatTensor(batch_feature).cuda()
            batch_iou = np.asarray(batch_iou_list)
            Y_iou = torch.FloatTensor(batch_iou).cuda()

            loss = run_pem(pem, X_feature, Y_iou)
            pem.zero_grad()
            loss.backward()
            optimizer.step()
            mini_info['iou_loss'].append(loss.data.cpu().numpy())
        train_info['iou_loss'].append(np.mean(mini_info['iou_loss']))

        # Validation
        pem.eval()
        mini_info = {'iou_loss':[], 'l2':[]}
        for idx in range(len(val_data)):
            prop_dict = val_data[idx]
            batch_feature, batch_iou_list, batch_ioa_list = PEM_load_data.prop_dict_data(prop_dict)

            X_feature = torch.FloatTensor(batch_feature).cuda()
            batch_iou = np.asarray(batch_iou_list)
            Y_iou = torch.FloatTensor(batch_iou).cuda()

            loss = run_pem(pem, X_feature, Y_iou)
            mini_info['iou_loss'].append(loss.data.cpu().numpy())
        val_info['iou_loss'].append(np.mean(mini_info['iou_loss']))

        print 'Epoch-{} Train Loss:  {:.04f}'.format(epoch, train_info['iou_loss'][-1] / 10)
        print 'Epoch-{} Val   Loss:  {:.04f}'.format(epoch, val_info["iou_loss"][-1] / 10)

        if val_info['iou_loss'][-1] < best_val_cost:
            best_val_cost = val_info['iou_loss'][-1]
            torch.save(pem.state_dict(), '{}/PEM/pem_model_best.pth'.format(opt.experiment))

    #
    #
    # data_root = '../../loss_test/pem'
    # loss_files = os.listdir(data_root)
    #
    # for loss_file in loss_files:
    #     file_path = os.path.join(data_root, loss_file)
    #     with open(file_path, 'rb') as input_file:
    #         loss_dict = cPickle.load(input_file)
    #
    #     match_iou = loss_dict['match_iou']
    #     anchors_iou = loss_dict['anchors_iou']
    #     iou_loss = loss_dict['iou_loss']
    #     u_smmask = loss_dict['u_smmask']
    #     u_slmask = loss_dict['u_slmask']
    #
    #     match_iou = torch.FloatTensor(match_iou).cuda()
    #     anchors_iou = torch.FloatTensor(anchors_iou).cuda()
    #
    #     loss = PEM_loss(anchors_iou, match_iou, u_smmask, u_slmask)
    #     print loss.data.cpu().numpy(), iou_loss
    #     assert (loss.data.cpu().numpy() - iou_loss) < 0.0001
