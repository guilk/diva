# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:21:39 2017

@author: wzmsltw
"""
import sys
sys.path.append('./Evaluation')
from eval_proposal import ANETproposal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def run_evaluation(ground_truth_filename, proposal_filename, 
                   max_avg_nr_proposals=100, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    
    return (average_nr_proposals, average_recall, recall)

def plot_metric(average_nr_proposals, average_recall, recall, dst_path, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    fn_size = 14
    fig = plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1,1,1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)

    # fig.savefig('./result.png')
    fig.savefig(dst_path)
    # plt.show()
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--splittype', default=None, help='split type (train, validation, test)')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_arguments()
    # eval_file="../../output/result_proposal.json"
    eval_file=os.path.join('../../output', opt.experiment, '{}_result_proposal.json'.format(opt.splittype))
    dst_path = os.path.join('../../output', opt.experiment, '{}_result.png'.format(opt.splittype))

    # uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
    #     "./Evaluation/data/activity_net_1_3_new.json",
    #     eval_file,
    #     max_avg_nr_proposals=100,
    #     tiou_thresholds=np.linspace(0.5, 0.95, 10),
    #     subset='validation')
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        "./Evaluation/data/virat_stride100_interval_300.json",
        eval_file,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset=opt.splittype)

    plot_metric(uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid, dst_path)

    #mean_100=np.mean(uniform_recall_valid[:,-1])
    print "AR@1 is \t",np.mean(uniform_recall_valid[:,0])
    print "AR@5 is \t",np.mean(uniform_recall_valid[:,4])
    print "AR@10 is \t",np.mean(uniform_recall_valid[:,9])
    print "AR@100 is \t",np.mean(uniform_recall_valid[:,-1])