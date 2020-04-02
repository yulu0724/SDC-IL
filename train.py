# coding=utf-8
from __future__ import absolute_import, print_function
from copy import deepcopy
import argparse
import os
import sys
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging, display
from torch.optim.lr_scheduler import StepLR
import pdb
import numpy as np
from ImageFolder import *
import torchvision.transforms as transforms
from evaluations import extract_features, pairwise_distance
from CIFAR100 import CIFAR100


cudnn.benchmark = True


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def fisher_matrix_diag(model, criterion, train_loader, number_samples=500):

    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0*p.data

    model.train()
    count = 0
    for i, data in enumerate(train_loader, 0):
        count += 1
        inputs, labels = data
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels).cuda()

        # Forward and backward
        model.zero_grad()
        if args.method == 'MAS':
            embed_feat = model.forward_without_norm(inputs)
            loss = torch.sum(torch.norm(embed_feat, 2, dim=1))
        elif args.method == 'EWC':
            _, embed_feat = model(inputs)
            if args.loss == 'MSLoss':
                loss = criterion(embed_feat, labels)
            else:
                loss, _, _, _ = criterion(embed_feat, labels)

        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    for n, _ in model.named_parameters():
        fisher[n] = fisher[n]/float(count)
        fisher[n] = torch.autograd.Variable(fisher[n], requires_grad=False)
    return fisher


def compute_prototype(model, data_loader):

    model.eval()
    count = 0
    embeddings = []
    embeddings_labels = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            count += 1
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            _, embed_feat = model(inputs)
            embeddings_labels.append(labels.numpy())
            embeddings.append(embed_feat.cpu().numpy())

    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(
        embeddings, (embeddings.shape[0]*embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(
        embeddings_labels, embeddings_labels.shape[0]*embeddings_labels.shape[1])
    labels_set = np.unique(embeddings_labels)
    class_mean = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        embeddings_tmp = embeddings[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_std.append(np.std(embeddings_tmp, axis=0))
    prototype = {'class_mean_old': class_mean, 'class_mean': class_mean,
                 'class_std': class_std, 'class_label': class_label}

    return prototype


def train_fun(args, train_loader, feat_loader, current_task, fisher={}, prototype={}):

    log_dir = os.path.join('checkpoints', args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))
    display(args)

    model = models.create(args.net, Embed_dim=args.dim)
    # load part of the model
    if args.method == 'Independent' or current_task == 0:
        model_dict = model.state_dict()

        if args.net == 'resnet32':
            if args.base == 50:
                pretrained_dict = torch.load(
                    'pretrained_models/Finetuning_0_task_0_200_model_task2_cifar100_seed1993.pkl')
            pretrained_dict = pretrained_dict.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items(
            ) if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        elif args.net == 'resnet18' and args.data == 'imagenet_sub':
            if args.base == 50:
                pretrained_dict = torch.load(
                    'pretrained_models/Finetuning_0_task_0_200_model_task2_imagenet_sub_seed1993.pkl')
            pretrained_dict = pretrained_dict.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items(
            ) if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        else:
            print (' Oops!  That was no valid models. ')

    if args.method != 'Independent' and current_task > 0:
        model = torch.load(os.path.join(log_dir, args.method + '_' + args.exp +
                                        '_task_' + str(current_task-1) + '_%d_model.pkl' % int(args.epochs-1)))
        model_old = deepcopy(model)
        model_old.eval()
        model_old = freeze_model(model_old)

    model = model.cuda()
    torch.save(model, os.path.join(log_dir, args.method + '_' +
                                   args.exp + '_task_' + str(current_task) + '_pre_model.pkl'))
    print('initial model is save at %s' % log_dir)

    # fine tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.Embed.parameters()))

    new_params = [p for p in model.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.parameters() if
                   id(p) not in new_param_ids]
    param_groups = [
        {'params': base_params, 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    criterion = losses.create(args.loss, margin=args.margin, num_instances=args.num_instances).cuda()
    optimizer = torch.optim.Adam(
        param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    if args.data == 'cifar100' or args.data == 'imagenet_sub':
        if current_task > 0:
            model.eval()

    for epoch in range(args.start, args.epochs):

        running_loss = 0.0
        running_lwf = 0.0
        scheduler.step()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            _, embed_feat = model(inputs)

            if current_task == 0:
                loss_aug = 0*torch.sum(embed_feat)
            else:
                if args.method == 'Finetuning' or args.method == 'Independent':
                    loss_aug = 0*torch.sum(embed_feat)
                elif args.method == 'LwF':
                    _, embed_feat_old = model_old(inputs)
                    loss_aug = args.tradeoff * \
                        torch.sum((embed_feat-embed_feat_old).pow(2))/2.
                elif args.method == 'EWC' or args.method == 'MAS':
                    loss_aug = 0
                    for (name, param), (_, param_old) in zip(model.named_parameters(), model_old.named_parameters()):
                        loss_aug += args.tradeoff * \
                            torch.sum(fisher[name]*(param_old-param).pow(2))/2.

            embed_sythesis = []
            embed_label_sythesis = []

            if args.loss == 'MSLoss':
                loss = criterion(embed_feat, labels)
                inter_ = 0
                dist_ap = 0
                dist_an = 0
            else:
                loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)
            loss += loss_aug

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            running_lwf += loss_aug.data[0]
            if epoch == 0 and i == 0:
                print(50*'#')
                print('Train Begin -- HA-HA-HA')

        print('[Epoch %05d]\t Total Loss: %.3f \t LwF Loss: %.3f \t Accuracy: %.3f \t Pos-Dist: %.3f \t Neg-Dist: %.3f'
              % (epoch + 1,  running_loss, running_lwf, inter_, dist_ap, dist_an))

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(log_dir, args.method + '_' +
                                           args.exp + '_task_' + str(current_task) + '_%d_model.pkl' % epoch))

    if args.method == 'EWC' or args.method == 'MAS':
        fisher = fisher_matrix_diag(
            model, criterion, train_loader, number_samples=500)
        return fisher


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='KNN-Softmax Training')

    # hype-parameters
    parser.add_argument('-lr', type=float, default=1e-4,
                        help="learning rate of new parameters")
    parser.add_argument('-tradeoff', type=float, default=1.0,
                        help="learning rate of new parameters")
    parser.add_argument('-exp', type=str, default='exp1',
                        help="learning rate of new parameters")
    parser.add_argument('-margin', type=float, default=0.0,
                        help="margin for metric loss")

    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('-num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('-dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('-alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in KNN Softmax')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')

    # network
    parser.add_argument('-data', default='cub', required=True,
                        help='path to Data Set')
    parser.add_argument('-net', default='bn')
    parser.add_argument('-loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='epochs for training process')

    parser.add_argument('-seed', default=1993, type=int, metavar='N',
                        help='seeds for training process')
    parser.add_argument('-save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')
    parser.add_argument('-lr_step', default=200, type=int, metavar='N',
                        help='number of epochs to save model')
    # Resume from checkpoint
    parser.add_argument('-start', default=0, type=int,
                        help='resume epoch')

    # basic parameter
    parser.add_argument('-log_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument("-gpu", type=str, default='0',
                        help='which gpu to choose')
    parser.add_argument("-method", type=str,
                        default='Finetuning', help='Choose FT or SC')

    parser.add_argument('-mapping_mean', default='no',
                        type=str, help='mapping')
    parser.add_argument('-sigma', default=0.0, type=float, help='sigma')
    parser.add_argument('-vez', default=0, type=int, help='vez')
    parser.add_argument('-task', default=0, type=int, help='vez')
    parser.add_argument('-base', default=50, type=int, help='vez')

    # parser.add_argument('-evel', action='store_true')

    args = parser.parse_args()
    # Data
    print('==> Preparing data..')

    if args.data == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        root = 'DataSet'
        traindir = root + '/cifar'
        num_classes = 100

    if args.data == 'cub':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/CUB_200_2011'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 200

    if args.data == 'car':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/Car196'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 196

    if args.data == 'flower':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/flowers'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 102

    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        root = '/datatmp/datasets/ILSVRC12_256'
        traindir = os.path.join(root, 'train')
        num_classes = 100

    num_task = args.task
    num_class_per_task = (num_classes-args.base)/(num_task-1)

    np.random.seed(args.seed)
    random_perm = np.random.permutation(num_classes)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fisher = {}
    prototype = {}
    for i in range(num_task):
        if i == 0:
            class_index = random_perm[:args.base]
        else:
            class_index = random_perm[args.base +
                                      (i-1)*num_class_per_task:args.base+i*num_class_per_task]
        if args.data == 'cifar100':
            trainfolder = CIFAR100(
                root=traindir, train=True, download=True, transform=transform_train, index=class_index)
        else:
            trainfolder = ImageFolder(
                traindir, transform_train, index=class_index)

        train_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=args.BatchSize,
            sampler=RandomIdentitySampler(
                trainfolder, num_instances=args.num_instances),
            drop_last=True, num_workers=args.nThreads)

        feat_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=1, shuffle=True, drop_last=False)
        # Fix the random seed to be sure we have the same permutation for one experiment
        if args.method == 'EWC' or args.method == 'MAS':
            fisher = train_fun(args, train_loader,
                               feat_loader, i, fisher=fisher)
        else:
            train_fun(args, train_loader, feat_loader, i)
