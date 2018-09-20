import torch
import torch.optim as optim

from torch.autograd import Variable
import torch.nn as nn
from TEM_model import TEM
import TEM_load_data
import argparse
import numpy as np

import os
import cPickle


def binary_logistic_loss(gt_scores, pred_anchors):
    '''
    Calculate weighted binary logistic loss
    :param gt_scores:
    :param pred_anchors:
    :return:
    '''
    # print gt_scores.size()
    # print pred_anchors.size()
    gt_scores = gt_scores.contiguous().view(-1)
    pred_anchors = pred_anchors.contiguous().view(-1)
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

def run_tem(tem_model, X_feature, Y_action, Y_start, Y_end):
    anchors = tem_model(X_feature)

    anchors_action = anchors[:,0,:]
    anchors_start = anchors[:,1,:]
    anchors_end = anchors[:,2,:]

    loss_action, action_num_sample = binary_logistic_loss(Y_action, anchors_action)
    loss_start, start_num_sample = binary_logistic_loss(Y_start, anchors_start)
    loss_end, end_num_sample = binary_logistic_loss(Y_end, anchors_end)
    loss = 2 * loss_action + loss_start + loss_end
    return loss, loss_action, loss_start, loss_end


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_arguments()
    num_epoches = opt.niter
    batch_size = opt.batchSize
    if opt.experiment == None:
        opt.experiment = './pytorch_models'

    # Intialize model
    tem = TEM()
    tem.cuda()

    optimizer = optim.Adam(tem.parameters(), lr = 0.001, betas=(opt.beta1, 0.999), weight_decay = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_dict, val_dict, test_dict = TEM_load_data.getDatasetDict()

    # small toy set for fast debugging
    # toy_dict = {}
    # for idx, (k,v) in enumerate(val_dict.iteritems()):
    #     if idx > 200:
    #         break
    #     toy_dict[k] = v
    # val_dict = toy_dict

    val_data_dict = TEM_load_data.getFullData(train_dict, val_dict, test_dict, "val")
    train_data_dict=TEM_load_data.getFullData(train_dict, val_dict, test_dict, "train")

    train_info={"loss":[],"loss_action":[],"loss_start":[],"loss_end":[]}
    val_info={"loss":[],"loss_action":[],"loss_start":[],"loss_end":[]}
    info_keys=train_info.keys()
    best_val_cost = 1000000


    for epoch in range(num_epoches):
        '''
        Train
        '''
        scheduler.step()
        # batch_video_list = TEM_load_data.getBatchList(len(val_dict), batch_size)
        batch_video_list = TEM_load_data.getBatchList(len(train_dict), batch_size)
        mini_info = {'loss':[], 'loss_action':[], 'loss_start':[], 'loss_end':[]}

        for p in tem.parameters():
            p.requires_grad = True
        tem.train()

        for idx in range(len(batch_video_list)):
            # print 'Process {}th batch'.format(idx)
            batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=\
                TEM_load_data.getBatchData(batch_video_list[idx],train_data_dict)
            # batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=\
            #     TEM_load_data.getBatchData(batch_video_list[idx],val_data_dict)

            batch_anchor_feature = np.transpose(batch_anchor_feature, (0, 2, 1))

            X_feature = torch.FloatTensor(batch_anchor_feature).cuda()
            Y_action = torch.FloatTensor(batch_label_action).cuda()
            Y_start = torch.FloatTensor(batch_label_start).cuda()
            Y_end = torch.FloatTensor(batch_label_end).cuda()

            loss, loss_action, loss_start, loss_end = run_tem(tem, X_feature, Y_action, Y_start, Y_end)
            mini_info['loss_action'].append(loss_action.data.cpu().numpy())
            mini_info['loss_start'].append(loss_start.data.cpu().numpy())
            mini_info['loss_end'].append(loss_end.data.cpu().numpy())
            mini_info['loss'].append(loss.data.cpu().numpy())
            tem.zero_grad()
            loss.backward()
            optimizer.step()
        train_info['loss_action'].append(np.mean(mini_info['loss_action']))
        train_info['loss_start'].append(np.mean(mini_info['loss_start']))
        train_info['loss_end'].append(np.mean(mini_info['loss_end']))

        '''
        Validation
        '''
        # for p in tem.parameters():
        #     p.requires_grad = True
        tem.eval()
        batch_video_list = TEM_load_data.getBatchList(len(val_dict), batch_size)
        mini_info = {'loss':[], 'loss_action':[], 'loss_start':[], 'loss_end':[]}
        for idx in range(len(batch_video_list)):
            batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=\
                TEM_load_data.getBatchData(batch_video_list[idx],val_data_dict)

            batch_anchor_feature = np.transpose(batch_anchor_feature, (0, 2, 1))
            X_feature = torch.FloatTensor(batch_anchor_feature).cuda()
            Y_action = torch.FloatTensor(batch_label_action).cuda()
            Y_start = torch.FloatTensor(batch_label_start).cuda()
            Y_end = torch.FloatTensor(batch_label_end).cuda()
            loss, loss_action, loss_start, loss_end = run_tem(tem, X_feature, Y_action, Y_start, Y_end)
            mini_info['loss_action'].append(loss_action.data.cpu().numpy())
            mini_info['loss_start'].append(loss_start.data.cpu().numpy())
            mini_info['loss_end'].append(loss_end.data.cpu().numpy())
            mini_info['loss'].append(loss.data.cpu().numpy())
        val_info['loss_action'].append(np.mean(mini_info['loss_action']))
        val_info['loss_start'].append(np.mean(mini_info['loss_start']))
        val_info['loss_end'].append(np.mean(mini_info['loss_end']))
        val_info['loss'].append(np.mean(mini_info['loss']))


        print 'Epoch-{} Train Loss: Action - {:.2f}, Start - {:.2f}, ' \
              'End - {:.2f}'.format(epoch, train_info['loss_action'][-1], train_info['loss_start'][-1], train_info['loss_end'][-1])
        print 'Epoch-{} Val Loss: Action - {:.2f}, Start - {:.2f}, ' \
              'End - {:.2f}'.format(epoch, val_info['loss_action'][-1], val_info['loss_start'][-1], val_info['loss_end'][-1])

        if val_info['loss'][-1] < best_val_cost:
            best_val_cost = val_info['loss'][-1]
            torch.save(tem.state_dict(), '{}/TEM/tem_model_best.pth'.format(opt.experiment))



    #
    # data_root = '../../loss_test/tem'
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
    #     print anchors_action.shape
    #     print Y_action.shape
    #     assert False
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

