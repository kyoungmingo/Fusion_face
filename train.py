import os
import numpy as np
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from fusion_dataset import Dataset

##data augmentation 수정##
# from rgbaug_dataset import Dataset

##data fusion##
from dataset import Dataset

from PIL import Image
from torch.utils import data as torch_data
from focal_loss import FocalLoss
from backbone.cbam import *
from backbone.resnet import *
from backbone.backbone_attention import *
from metrics import *
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from datasplit import datasplit
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from collections import Counter
from torch.utils.data.dataset import random_split
import cv2
from torchvision import transforms as T

def save_model(model, save_path, name, iter_cnt=None):
    if iter_cnt is not None:
        save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    else:
        save_name = os.path.join(save_path, name + '_'  + '.pth')
    torch.save(model.state_dict(), save_name)

    return save_name

def imshow(img,one_ch = False):
    if one_ch:
        img = img.mean(dim=0)
    img = img/2 + 0.5
    npimg = img.numpy()
    if one_ch:
        plt.imshow(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(np.transpose(npimg,(1,2,0)))


def train(opt):

    ##valid data 추가 0401####################################
    # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
    #                         std=[0.5, 0.5, 0.5])
    # valid_transforms = T.Compose([
    #     T.ToTensor(),
    #     T.CenterCrop((128, 128)),
    #     normalize
    # ])
    # my_path = "/mnt/nas2/kkm/face_recognition/testcrop_resize_"
    # img = [my_path + str(i) + '.jpg' for i in range(1, 16)]
    # for i in img:
    #     data = Image.open(i)
    #     b, g, r = data.split()
    #     data = Image.merge("RGB", (r, g, b))
    #     data = valid_transforms(data)
    #     data = data.unsqueeze(0)
    #     if i == '/mnt/nas2/kkm/face_recognition/testcrop_resize_1.jpg':
    #         fdata = data
    #     else:
    #         fdata = torch.cat((fdata, data), dim=0)
    #
    # real_ans = [5, 5, 8, 8, 8, 8, 8, 7, 7, 1, 1, 2, 2, 0, 0]
    ##########################################################


    # best_train_acc = 0.0
    best_test_acc = 0.0

    final_loss = 1.0

    easy_margin = False
    num_classes = 83
    save_interval = 1
    optimizer_type = 'adam'
    num_workers = 8  # how many workers for loading data
    max_epoch = opt.epoch
    lr = 2e-4  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.1

    backbone = opt.backbone


    use_se = False

    # if opt.display:
    #     visualizer = Visualizer()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #tensorboard path
    writer = SummaryWriter(opt.runs)
    print('tensorboard :',opt.runs)
    print('feature type : ',opt.feature_type)
    print('train_nums : ',opt.train_nums)
    print('train batch_size : ',opt.train_batch_size)

    if opt.train_nums==1 :
        opt.train_root="/mnt/nas2/hm/hm_fusionFR/data/data_num1/train"
        opt.test_root="/mnt/nas2/hm/hm_fusionFR/data/data_num1/test"
    elif opt.train_nums==3 :
        opt.train_root="/mnt/nas2/hm/hm_fusionFR/data/data_num3/train"
        opt.test_root="/mnt/nas2/hm/hm_fusionFR/data/data_num3/test"
    elif opt.train_nums==5 :
        opt.train_root="/mnt/nas2/hm/hm_fusionFR/data/data_num5/train"
        opt.test_root="/mnt/nas2/hm/hm_fusionFR/data/data_num5/test"
    elif opt.train_nums==7  :
        opt.train_root="/mnt/nas2/hm/hm_fusionFR/data/data_num7/train"
        opt.test_root="/mnt/nas2/hm/hm_fusionFR/data/data_num7/test"
    else:
        ##data augmentation 수정##
        ##AIhub + our##
        # opt.train_root = "/mnt/nas2/kkm/face_recognition/facedata/train"
        # opt.test_root = "/mnt/nas2/kkm/face_recognition/facedata/test"

        ##5 shot##
        # opt.train_root = '/mnt/nas2/hm/hm_fusionFR/data/mtcnncropresize/data_num5/train'
        # opt.test_root = '/mnt/nas2/hm/hm_fusionFR/data/mtcnncropresize/data_num5/test'

        ##only AIhub for pretrained weight##
        # opt.train_root = '/mnt/nas1/k_kkm/face_recognition/train_data'

        ##fusion crop data##
        opt.train_root = "/mnt/nas1/k_hm3346/hm/hm_fusionFR/newdata/shot3/train"
        opt.test_root = "/mnt/nas1/k_hm3346/hm/hm_fusionFR/newdata/shot3/test"

    print('train data path : ', opt.train_root)

    if opt.test_root is not None:
        train_dataset = Dataset(opt.train_root, phase='train',feature_type=opt.feature_type)
        test_dataset = Dataset(opt.test_root, phase='test',feature_type=opt.feature_type)

        trl=len(train_dataset)
        tll=len(test_dataset)
        print('train length : ',trl)
        print('test length : ',tll)

        trainloader = torch_data.DataLoader(train_dataset,
                                      batch_size=opt.train_batch_size,
                                      shuffle=True,
                                      num_workers = num_workers,pin_memory = True
                                            ,drop_last = True)

        # test_batchsize=len(test_dataset)
        test_batchsize=opt.test_batch_size

        testloader = torch_data.DataLoader(test_dataset,
                                            batch_size=test_batchsize,
                                            shuffle=True,
                                            num_workers=num_workers,pin_memory = True
                                           ,drop_last = True)


    ##test root default를 None으로 수정하였음##
    else:

        train_dataset, test_dataset =random_split(Dataset(opt.train_root, phase='train',feature_type=opt.feature_type), [2241, 560])

        trl = len(train_dataset)
        tll = len(test_dataset)

        trainloader = torch_data.DataLoader(train_dataset,
                                            batch_size=opt.train_batch_size,
                                            shuffle=True,
                                            num_workers=num_workers, pin_memory=True,
                                            drop_last=True)

        testloader = torch_data.DataLoader(test_dataset,
                                           batch_size=opt.test_batch_size,
                                           shuffle=True,
                                           num_workers=num_workers, pin_memory=True,
                                           drop_last=True)



    # identity_list = get_lfw_list(opt.test_list)
    # img_paths = [os.path.join(opt.test_root, each) for each in identity_list]


    print('{} train iters per epoch:'.format(len(trainloader)))
    print('{} train iters per epoch:'.format(len(testloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.feature_type == 'rgb':
        num_chanels = 3
    elif opt.feature_type == 'depth':
        num_chanels = 4
    elif opt.feature_type == 'points':
        num_chanels = 6
    elif opt.feature_type == 'full':
        num_chanels = 7

    # num_chanels = 7 if opt.feature_type == 'full' else 3
    print('num_chanels : ',num_chanels)

    if backbone == 'resnet18':
        model = resnet_face18(num_chanels=num_chanels,use_se=use_se,feature_dim=512)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=True)
    elif backbone == 'resnet50':
        model = resnet50()
    elif backbone == 'CBAM':
        # CBAMResNet(layer,dim,mode)
        model = CBAMResNet(50, feature_dim=512, mode='ir_se')
    elif backbone == 'SAC':
        model = Backbone_pt()
    else:
        model = resnetfusion(pretrained=opt.pretrained,num_chanels=num_chanels)
    print(backbone)


    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.35, easy_margin=easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, num_classes, m=4)
    else:
        # metric_fc = nn.Linear(512, num_classes)
        metric_fc = InnerProduct(512, num_classes)

    print(opt.metric)

    # view_model(model, opt.input_shape)
    if backbone not in  ['fusion']:
        model.to(device)
        model = DataParallel(model)
        print(model)

        if opt.pretrained is not None:
            model.load_state_dict(torch.load(opt.pretrained))
            print('success pretrained weight')

    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    #fc layer를 가져오기 위함
    if opt.pretrained_metric is not None:
        pretrained = torch.load(opt.pretrained_metric)
        weight_init = Parameter(torch.Tensor(483, 512))
        nn.init.xavier_uniform_(weight_init)
        pretrained['module.weight'] = weight_init
        metric_fc.load_state_dict(pretrained)
        print('success pretrained metric')



    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=lr, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)


    print()
    print('train length : ',len(train_dataset))
    print('test length : ',len(test_dataset))
    print('TRAIN START !! ')
    start = time.time()
    for i in range(max_epoch):

        # scheduler.step()

        total_loss = []
        total_acc = []
        total_test_loss = []

        train_pred=[]
        train_real=[]

        model.train()

        epoch_starttime = time.time()
        train_corrects = 0.0
        for ii, my_data in enumerate(trainloader):

            data_input, label = my_data


            data_input = data_input.to(device)
            label = label.to(device).long()

            feature = model(data_input)
            # print(feature.shape)

            #####0324수정#####
            if opt.metric not in ['add_margin','arc_margin','sphere']:
                #softmax 시에는 label이 필요없음.
                output = metric_fc(feature)
            else:
                output = metric_fc(feature, label)
            
            #output = metric_fc(feature)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)

            label = label.data.cpu().numpy()

            total_loss.append(loss.item())

            # print('loss',loss)

            acc = np.mean((output == label).astype(int))
            train_corrects += (output==label).sum()

            total_acc.append(acc)

            train_pred.append(output)
            train_real.append(label)

        print('train pred : ',train_pred)
        print('train real value : ',train_real)


        # if opt.test_root=='/mnt/nas2/hm/hm_fusionFR/data/splitdata/test' :
        test_corrects = 0.0
        test_wrong = []
        model.eval()
        for j,my_test in enumerate(testloader):
            test_input, test_label = my_test

            test_input = test_input.to(device)
            test_label = test_label.to(device).long()

            test_input = model(test_input)

            #########0324 수정##########
            test_output = metric_fc(test_input)

            test_loss = criterion(test_output, test_label)
            total_test_loss.append(test_loss.item())

            # if opt.metric not in ['add_margin','arc_margin','sphere']:
            #     #softmax 시에는 label이 필요없음.
            #     test_output = metric_fc(test_input)
            # else:
            #     test_output = metric_fc(test_input)
            ############################

            test_output = test_output.data.cpu().numpy()
            test_output = np.argmax(test_output, axis=1)

            test_label = test_label.data.cpu().numpy()
            test_corrects += (test_output==test_label).sum()
            for jj in range(len(test_label)):
                if test_output[jj] != test_label[jj]:
                    test_wrong.append(test_label[jj])

            print('test label : ',test_label)
            print('test output : ',test_output)

        #############valid추가#############
        # valid_feature = model(fdata)
        # vaild_output = metric_fc(valid_feature)
        # vaild_output = np.argmax(vaild_output.cpu().detach().numpy(), axis=1)
        #
        # res = [real_ans[i] == vaild_output[i] for i in range(len(real_ans))]
        #############valid추가#############

        print('train nums : ', opt.train_nums)

        print('train length : ', len(train_dataset))
        print('test length : ', len(test_dataset))

        print('test wrong label : ', sorted(test_wrong, reverse=False))
        print('Counter - test wrong label : ', Counter(test_wrong))

        print("epoch ", i, " train corrects : ", train_corrects)
        print("epoch ", i, " test corrects : ", test_corrects)

        print("epoch ", i, ' test accuracy(total) : ', round(((test_corrects) / tll) * 100, 7), " % ")
        print("epoch ", i, ' train accuracy(total) : ', round(((train_corrects) / trl) * 100, 7), " % ")

        print('train epoch {}  loss {} acc : {}'.format(i, np.array(total_loss).mean(), np.array(total_acc).mean()))

        print('running time by one epoch : ', time.time() - epoch_starttime, ' seconds')
        # print('valid real: ', real_ans)
        # print('valid output: ', vaild_output)
        # print('valid acc: ', sum(res))

            # batch size를 줄이던, test시에 결과를 확인할 수 있음.
            # writer.add_figure('visualize',plot_classes_preds(output,data_input,label),
            #         global_step=iters)

            # iters = i * len(trainloader) + ii
            # data_input = data_input.data.cpu()
        # test_acc = lfw_test(model, img_paths, identity_list, opt.test_list, opt.test_batch_size, device=device)

        writer.add_scalar('train loss', np.array(total_loss).mean(), i)
        writer.add_scalar('test loss', np.array(total_test_loss).mean(), i)
        writer.add_scalar('train acc', round(((train_corrects) / trl),7  ), i)
        writer.add_scalar('test acc', round(((test_corrects) / tll),7  ), i)
        # writer.add_scalar('valid acc',sum(res), i)

        if i % save_interval == 0 or i == max_epoch:
            if not os.path.exists(opt.checkpoints_path):
                os.mkdir(opt.checkpoints_path)
            save_model(model, opt.checkpoints_path, backbone, i)
            save_model(metric_fc, opt.checkpoints_path, opt.metric, i)

        if final_loss > np.array(total_loss).mean():
            print("Best Loss!!!")
            final_loss=np.array(total_loss).mean()
            save_model(model, opt.checkpoints_path, backbone +'train')
            save_model(metric_fc, opt.checkpoints_path, opt.metric + 'train')

        if test_corrects > best_test_acc:
            print("Best mean acc in test set")
            best_test_acc = test_corrects
            save_model(model, opt.checkpoints_path, backbone +'test')
            save_model(metric_fc, opt.checkpoints_path, opt.metric + 'test')

    # model = resnet_face18(use_se=opt.use_se)
    # model = DataParallel(model)
    # model.load_state_dict(torch.load(opt.test_model_path))
    # model.to(torch.device("cuda"))
    # acc = lfw_test(model, img_paths, identity_list, opt.test_list, opt.test_batch_size,divices=device)
    #     # writer.add_scalar('Test acc', np.array(acc).mean(), i)
    # if opt.display:
    #     visualizer.display_current_results(iters, acc, name='test_acc')

    # shutil.rmtree('/mnt/nas2/hm/hm_fusionFR/data/splitdata/train', ignore_errors=True)
    # shutil.rmtree('/mnt/nas2/hm/hm_fusionFR/data/splitdata/test', ignore_errors=True)



if __name__ == '__main__':

    # parser = argparse.ArgumentParser("./train_go.py")
    parser = argparse.ArgumentParser(description = 'Argparse')

    parser.add_argument(
      '--train_root', '-train',
      type=str,
      # default ='/mnt/nas2/kkm/face_recognition/kface',
      # default='/mnt/nas2/kkm/face_recognition/train_point',
        default='/mnt/nas2/hm/hm_fusionFR/data/splitdata/train',
        help='Train dataset path'
    )
    parser.add_argument(
      '--test_root', '-test',
      type=str,
      # default='test_img',
        default= None,
        help='Test dataset path'
    )
    parser.add_argument(
      '--test_list', '-testpair',
      type=str,
      default='test_list_koleesim.txt',
      help='Test pair list to test with. No Default'
    )
    parser.add_argument(
        '--backbone', '-backbone',
        type=str,
        default='resnet18',
        help='backbone network'
    )
    #model은 직접 parameter를 수정하도록함.
    # parser.add_argument(
    #     '--model', '-m',
    #     type=str,
    #     default=,
    #     help='Directory to put the log data. Default: ~/logs/date+time'
    # )
    parser.add_argument(
      '--feature_type', '-ftype',
      type=str,
      default='rgb',
      help='feature_type : RGB or FULL(RGB,Depth map,Point Cloud(X,Y,Z) '
    )
    parser.add_argument(
      '--train_nums', '-trn',
      type=int,
      default='3',
      help='train_nums : number of train data '
    )


    parser.add_argument(
      '--metric', '-m',
      type=str,
      default='add_margin',
      help='margin_type : add_margin,arc_margin,sphere_margin,softmax_margin'
    )
    parser.add_argument(
      '--loss', '-l',
      type=str,
      default='focal_loss',
      help='loss_type : focal_loss, CE_loss'
    )
    parser.add_argument(
      '--pretrained', '-p',
      type=str,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
    )

    parser.add_argument(
        '--pretrained_metric', '-pm',
        type=str,
        default=None,
        help='Directory to get the pretrained model. If not passed, do from scratch!'
    )
    parser.add_argument(
      '--checkpoints_path', '-check',
      type=str,
      default='checkpoints/arcpoint_0318_res',
      help='Directory to save the checkpoint.'
    )
    parser.add_argument(
      '--train_batch_size', '-trainb',
      type=int,
      default=32,
      help='train_batch_size'
    )
    parser.add_argument(
      '--test_batch_size', '-testb',
      type=int,
      default=64,
      help='test_batch_size'
    )
    parser.add_argument(
      '--gpu', '-gpu',
      type=str,
      default=True,
      help='Use gpu!'
    )
    parser.add_argument(
        '--epoch', '-epoch',
        type=int,
        default=10,
        help='Use epoch!'
    )
    parser.add_argument(
        '--runs', '-runs',
        type=str,
        default='runs/0325',
        help='Use epoch!'
    )

    args = parser.parse_args()

    train(args)






#
# #test 시, 결과 Visualize
# #
#
# train_dataset = Dataset('/mnt/nas2/kkm/face_recognition/train_point', phase='train', input_shape=(7, 256, 192))
# trainloader = dt.DataLoader(train_dataset,
#                                   batch_size=1,
#                                   shuffle=True,
#                                   num_workers = 4)
#
# data,label = next(iter(trainloader))
#
# # model = resnet_face18(use_se=use_se)
# model = Backbone_pt()
#
# model = DataParallel(model)
# model.cuda()
# model.load_state_dict(torch.load('/mnt/nas2/kkm/face_recognition/arcface-pytorch/checkpoints/arcpoint_0309/SAC_140.pth'))
#
# model.eval()
# data = data.cuda()
#
# output = model(data)
#
# label=label.cuda()
#
# # def plot_classes_preds(output, images, labels):
# #
# #     # preds, probs = images_to_probs(net, images)
# #     # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
# #     fig = plt.figure(figsize=(12, 48))
# #     for idx in np.arange(images.shape[0]):
# #         ax = fig.add_subplot(1, images.shape[0], idx+1, xticks=[], yticks=[])
# #         imshow(images[idx], one_ch=True)
# #         ax.set_title("pred : {},\nlabel: {}".format(
# #             output[idx],
# #             labels[idx]),
# #                     color=("green" if output[idx]==labels[idx].item() else "red"))
# #     return fig
# #
# # plot_classes_preds(output.cpu(),data.cpu(),label.cpu())
#
# from sklearn.manifold import TSNE
#
# X_embedded = TSNE(n_components=2).fit_transform(output.cpu().detach().numpy())
#
# plt.figure(figsize=(10,10))
#
# for i, t in enumerate(set(label.cpu().detach().numpy())):
#     idx = label.cpu().detach().numpy() == t
#     plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)
#
# # plt.legend(bbox_to_anchor=(1, 1))
#
# plt.show()
