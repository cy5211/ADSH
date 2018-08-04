#-*- coding:UTF-8 -*-
import utils.adsh_loss as al
import utils.data_processing as dp
import utils.cnn_model as cnn_model
import utils.subset_sampler as subsetsampler
import utils.calc_hr as calc_hr

import pickle
import os
import argparse
import logging
import torch
import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
import torch.nn as nn

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader



parser = argparse.ArgumentParser(description="ADSH demo")
#parser.add_argument('--bits', default='12,24,32,48', type=str,
#                   help='binary code length (default: 12,24,32,48)')
parser.add_argument('--bits', default='12', type=str,
                   help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='1', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=50, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--num-samples', default=2000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--num-triplet-samples', default=5900, type=int,
                    help='hyper-parameter: number of triplet samples (default: 1000)')
parser.add_argument(
    '--alpha',
    default=1,
    type=int,
    help='hyper-parameter: alpha (default: 1)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--lamda', default=0, type=int,
                    help='hyper-parameter: lamda (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument(
    '--model-save-path',
    default='/home/cy/ADSH_pytorch/ADSH_pytorch/model-cifar10',
    type=str,
    help=
    'model save path (default: /home/cy/ADSH_pytorch/ADSH_pytorch/model-10-labels)'
)

def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingCIFAR_10(
        'data/CIFAR-10', 'database_img.txt', 'database_label.txt', transformations)
    
    print 'dset_database',dset_database

    dset_test = dp.DatasetProcessingCIFAR_10(
        'data/CIFAR-10', 'test_img.txt', 'test_label.txt', transformations)
        
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))

    testlabels = load_label('test_label.txt', 'data/CIFAR-10')
    databaselabels = load_label('database_label.txt', 'data/CIFAR-10')

    testlabels = encoding_onehot(testlabels)
    databaselabels = encoding_onehot(databaselabels)

    print "cifar10 testlabel shape",testlabels.shape
    print "cifar10 trainlabel shape",databaselabels.shape

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels

#return S:query_num*train_num
def calc_sim(database_label, train_label):
    #大于0的置为1
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)

    '''
    soft constraint
    '''

    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r

    return S

#每次传入的U的数量为num_samples,这个函数用的是numpy操作,只在query data中统计三元损失.
def calc_loss_query(V, U, S, S_query, code_length, select_index, gamma,lamda):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    U_dot_U=U.dot(U.transpose())
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)

    sum_triplet_loss=0
    for i in range(opt.num_samples):
        #输出为turple
        _index_unsim = (S_query[i] < 0).nonzero()
        select_index_unsim = list(np.random.permutation(
            list(_index_unsim[0])))[0:opt.num_triplet_samples]

        _index_sim = (S_query[i] > 0).nonzero()
        select_index_sim = list(np.random.permutation(
            list(_index_sim[0])))[0:opt.num_triplet_samples]

        _temp=np.maximum((U_dot_U[i][select_index_unsim] - U_dot_U[i][select_index_sim]) / 2.0 +
            code_length / 2.0, 0)

        sum_triplet_loss += _temp.sum()

    loss+=lamda*sum_triplet_loss/(opt.num_samples*opt.num_triplet_samples)

    return loss

#每次传入的U的数量为num_samples,这个函数用的是numpy操作,用query data在database中统计三元损失.
def calc_loss(V, U, S, code_length, select_index, gamma,lamda):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    U_dot_V=U.dot(V.transpose())
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)

    sum_triplet_loss=0
    '''
    for i in range(opt.num_samples):
        #输出为turple
        _index_unsim = (S[i] < 0).nonzero()
        select_index_unsim = list(np.random.permutation(
            list(_index_unsim[0])))[0:opt.num_triplet_samples]

        _index_sim = (S[i] > 0).nonzero()
        select_index_sim = list(np.random.permutation(
            list(_index_sim[0])))[0:opt.num_triplet_samples]

        _temp=np.maximum((U_dot_V[i][select_index_unsim] - U_dot_V[i][select_index_sim]) / 2.0 +
            code_length / 2.0, 0)

        sum_triplet_loss += _temp.sum()

    loss+=lamda*sum_triplet_loss/(opt.num_samples*opt.num_triplet_samples)'''

    return square_loss.sum() / (
        opt.num_samples * num_database), quantization_loss.sum() / (
            opt.num_samples ), lamda * sum_triplet_loss / (
                opt.num_samples * opt.num_triplet_samples), loss


def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def adsh_algo(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    alpha = opt.alpha
    gamma = opt.gamma
    lamda = opt.lamda
    model_save_path = opt.model_save_path

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_length)
    model.cuda()
    adsh_loss = al.ADSHLoss(alpha, gamma, lamda, code_length, num_database)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))

    model.train()

    for iter in range(max_iter):
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)

        '''
        learning deep neural network: feature learning
        '''
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                #是一个batch_size
                train_input = Variable(train_input.cuda())

                output = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                S_query=calc_sim(sample_label[u_ind, :],sample_label[u_ind, :])
                U[u_ind, :] = output.cpu().data.numpy()

                model.zero_grad()
                loss = adsh_loss(output, V, S, S_query, V[batch_ind.cpu().numpy(), :],1)
                loss.backward()
                optimizer.step()

        #print optimizer.state_dict()
        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''

        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        iter_time = time.time() - iter_time
        S_query=calc_sim(sample_label,sample_label)
        square_loss_,quanty_loss_,triplet_loss_,loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma, lamda)
        logger.info('[Iteration: %3d/%3d][square Loss: %.4f][quanty Loss: %.4f][triplet Loss: %.4f][train Loss: %.4f]', iter, max_iter, square_loss_,quanty_loss_,triplet_loss_,loss_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

    '''
    training procedure finishes, evaluation
    '''

    torch.save(model, model_save_path)
    print "model saved!"

    model.eval()
    testloader = DataLoader(dset_test, batch_size=1,
                             shuffle=False,
                             num_workers=4)
    qB = encode(model, testloader, num_test, code_length)
    rB = V
    #计算有序性
    map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
    logger.info('[Evaluation: mAP: %.4f]', map)
    record['rB'] = rB
    record['qB'] = qB
    record['map'] = map
    filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)

def adsh_eval(code_length):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    model_save_path = opt.model_save_path

    #model = torch.load(model_save_path)
    
    inf=pickle.load(open('./log/log-ADSH-cifar10-18-07-30-10-15-54'))
    V=inf['rB']

    #model.eval()
    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset(10)
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels
    '''
    testloader = DataLoader(
        dset_test, batch_size=1, shuffle=False, num_workers=4)'''

    print "here!"
    #qB = encode(model, testloader, num_test, num_testing, code_length)
    #print "query shape",qB.shape

    qB = V[0:1000]
    rB = V
    #计算有序性
    map = calc_hr.calc_map(qB, rB,
                           database_labels.numpy()[0:1000],
                           database_labels.numpy())
    print map
    #logger.info('[Evaluation: mAP: %.4f]', map)

if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    '''logdir = '-'.join(['log/log-ADSH-cifar10', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()'''
    bits = [int(bit) for bit in opt.bits.split(',')]
    for bit in bits:
        adsh_eval(bit)
