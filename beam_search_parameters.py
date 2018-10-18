import os



def bsn_pipeline(tem_lr, tem_batchsize, tem_embedsize, tem_hiddensize, tem_stepsize, tem_gamma, pem_lr,
                 pem_hiddensize, pem_batchsize):
    experiment_type = 'tem_lr{}_bsize{}_emsize{}_hsize{}_stepsize{}_gamma{}_pem_lr{}_hsize{}_bsize{}'\
        .format(tem_lr, tem_batchsize, tem_embedsize, tem_hiddensize, tem_stepsize, tem_gamma, pem_lr,
                pem_hiddensize, pem_batchsize)

    print experiment_type

    cmd = 'mkdir -p ../../output/{}/PEM_results'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'mkdir -p ../../output/{}/TEM_results'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'mkdir -p ../../output/{}/PGM_proposals'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'mkdir -p ../../output/{}/PGM_features'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_TEM_train.py --lr {} --batchsize {} --embedsize {} --hiddensize {} --stepsize {} --gamma {} --experiment {}'\
        .format(tem_lr, tem_batchsize, tem_embedsize, tem_hiddensize, tem_stepsize, tem_gamma, experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_TEM_test.py --embedsize {} --hiddensize {} --experiment {}'\
        .format(tem_embedsize, tem_hiddensize, experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'python PGM_proposal_generation.py --experiment {}'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'python PGM_feature_generation.py --experiment {}'.format(experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_PEM_train.py --lr {} --hiddensize {} --batchsize {} --experiment {}'\
        .format(pem_lr, pem_hiddensize, pem_batchsize, experiment_type)
    os.system(cmd)
    # print cmd

    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_PEM_test.py --experiment {} --hiddensize {} --splittype {}'.format(experiment_type, pem_hiddensize, 'train')
    os.system(cmd)
    # print cmd
    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_PEM_test.py --experiment {} --hiddensize {} --splittype {}'.format(experiment_type, pem_hiddensize, 'validation')
    os.system(cmd)
    # print cmd
    cmd = 'CUDA_VISIBLE_DEVICES=1,2 python pytorch_PEM_test.py --experiment {} --hiddensize {} --splittype {}'.format(experiment_type, pem_hiddensize, 'test')
    os.system(cmd)
    # print cmd

    cmd = 'python Post_processing.py --experiment {} --splittype {}'.format(experiment_type, 'train')
    os.system(cmd)
    # print cmd
    cmd = 'python Post_processing.py --experiment {} --splittype {}'.format(experiment_type, 'validation')
    os.system(cmd)
    # print cmd
    cmd = 'python Post_processing.py --experiment {} --splittype {}'.format(experiment_type, 'test')
    os.system(cmd)
    # print cmd

    cmd = 'python eval.py --experiment {} --splittype {}'.format(experiment_type, 'train')
    os.system(cmd)
    # print cmd
    cmd = 'python eval.py --experiment {} --splittype {}'.format(experiment_type, 'validation')
    os.system(cmd)
    # print cmd
    cmd = 'python eval.py --experiment {} --splittype {}'.format(experiment_type, 'test')
    os.system(cmd)
    # print cmd
    # assert False

if __name__ == '__main__':

    tem_lrs = [0.005, 0.001, 0.0005, 0.0001]
    tem_batchsizes = [4, 8, 16, 32]
    tem_embedsizes = [32, 64, 128, 256, 512]
    tem_hiddensizes = [32, 64, 128, 256, 512]
    tem_stepsizes = [5, 10]
    tem_gammas = [0.1, 0.2, 0.5]

    pem_lrs = [0.001, 0.0005, 0.0001]
    pem_hiddensizes = [64, 128, 256]
    pem_batchsizes = [8, 16, 32]

    for tem_lr in tem_lrs:
        for tem_batchsize in tem_batchsizes:
            for tem_embedsize in tem_embedsizes:
                for tem_hiddensize in tem_hiddensizes:
                    for tem_stepsize in tem_stepsizes:
                        for tem_gamma in tem_gammas:

                            for pem_lr in pem_lrs:
                                for pem_hiddensize in pem_hiddensizes:
                                    for pem_batchsize in pem_batchsizes:
                                        bsn_pipeline(tem_lr, tem_batchsize, tem_embedsize,
                                                     tem_hiddensize, tem_stepsize, tem_gamma, pem_lr, pem_hiddensize,
                                                     pem_batchsize)