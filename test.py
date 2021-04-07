# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
# from config import Config
from torch.nn import DataParallel

# class Config(object):
#     env = 'default'
#     backbone = 'resnet18'
#     classify = 'softmax'
#     num_classes = 13938
#     metric = 'arc_margin'
#     easy_margin = False
#     use_se = False
#     loss = 'focal_loss'

#     display = False
#     finetune = False

#     train_root = '/mnt/hyun/face/arcface-pytorch/detect_img'
#     train_list = '/data/Datasets/webface/train_data_13938.txt'
#     val_list = '/data/Datasets/webface/val_data_13938.txt'

#     test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
#     test_list = 'test.txt'

#     lfw_root = '/mnt/hyun/face/arcface-pytorch/data/Datasets/lfw/lfw-align-128'
#     lfw_test_list = '/mnt/hyun/face/arcface-pytorch/test_list_연구실사람들.txt'

#     checkpoints_path = 'checkpoints'
#     load_model_path = 'models/resnet18.pth'
#     test_model_path = 'checkpoints/resnet18_110.pth'
#     save_interval = 10

#     train_batch_size = 16  # batch size
#     test_batch_size = 4

#     input_shape = (1, 128, 128)

#     optimizer = 'sgd'

#     use_gpu = True  # use GPU or not
#     gpu_id = '0, 1'
#     num_workers = 4  # how many workers for loading data
#     print_freq = 100  # print info every N batch

#     debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
#     result_file = 'result.csv'

#     max_epoch = 50
#     lr = 1e-1  # initial learning rate
#     lr_step = 10
#     lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
#     weight_decay = 5e-4


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = cv2.resize(image,(128,128))
    image = np.dstack((image, np.fliplr(image)))
    # print(image.shape)
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    # print(image.shape)
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    # print(image.shape)
    return image


def get_featurs(model, test_list, batch_size=4,device='cuda'):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))
        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        # print(images.shape)

            if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
                cnt += 1

                data = torch.from_numpy(images)
                print(data.shape)
                data = data.to(torch.device(device))
                output = model(data)
                output = output.data.cpu().numpy() # output -> (60,512)
                fe_1 = output[::2]
                fe_2 = output[1::2]
                # print(fe_1.shape, fe_2.shape)
                feature = np.hstack((fe_1, fe_2))
                # print(feature.shape)

                if features is None:
                    features = feature
                else:
                    features = np.vstack((features, feature))

                images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        try:
            fe_dict[each] = features[i]
        except:
            pass
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        try:
            fe_1 = fe_dict[splits[0]]
            fe_2 = fe_dict[splits[1]]
            label = int(splits[2])
            sim = cosin_metric(fe_1, fe_2)

            sims.append(sim)
            labels.append(label)
        except:
            pass

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size,divices):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size,device=divices)
    """
    features -> (5969, 1024)
    """
    t = time.time() - s
    # print('total time is {}, average time is {}'.format(t, t / cnt))

    fe_dict = get_feature_dict(identity_list, features)
    """
    {key : identity_path, value : feature vector}
    """
    acc, th = test_performance(fe_dict, compair_list)
    print('test accuracy : {} threshold : {}'.format(acc,th))
    return acc





if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)
    print(identity_list[:5])
    img_paths = [os.path.join(opt.train_root, each) for each in identity_list]
    print(img_paths[:5])
    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)



