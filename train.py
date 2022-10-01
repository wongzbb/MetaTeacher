from __future__ import print_function
import os
import sys
import time
import logging
import random
import argparse
from sklearn import metrics
import numpy as np
from timm.utils import AverageMeter
from visdom import Visdom
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from utils import *
from metric.loss import FitNet, AttentionTransfer, RKdAngle, RkdDistance
from data_list import ImageList, ImageList_idx

# Teacher models:
# VGG11/VGG13/VGG16/VGG19, GoogLeNet, AlxNet, ResNet18, ResNet34,
# ResNet50, ResNet101, ResNet152, ResNeXt29_2x64d, ResNeXt29_4x64d,
# ResNeXt29_8x64d, ResNeXt29_32x64d, PreActResNet18, PreActResNet34,
# PreActResNet50, PreActResNet101, PreActResNet152,
# DenseNet121, DenseNet161, DenseNet169, DenseNet201,
import models

# Student models:
# myNet, LeNet, FitNet

start_time = time.time()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch LR_adaptive_AT')

parser.add_argument('--dataset',
                    choices=['t1_chest_x_ray',
                             't2_chexpert',
                             's1_google_health',
                             's2_openi'
                             ],
                    default='s1_google_health')
parser.add_argument('--teachers',
                    choices=['ResNet32',
                             'ResNet44',
                             'ResNet50',
                             'ResNet56',
                             'ResNet110'
                             ],
                    default=['ResNet50', 'ResNet50', 'ResNet50'],
                    nargs='+')
parser.add_argument('--teachers_dir',
                    default=['chest_x_ray_0.771', 'chexpert_0.8135'],
                    nargs='+')
parser.add_argument('--student',
                    choices=['ResNet50',
                             'ResNet20',
                             'myNet'
                             ],
                    default='ResNet50')

parser.add_argument('--kd_ratio', default=0.7, type=float)
parser.add_argument('--n_class', type=int, default=6, metavar='N', help='num of classes')
parser.add_argument('--T', type=float, default=20.0, metavar='Temputure', help='Temputure for distillation')
parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=2, metavar='N', help='input test batch size for training')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--device', default='cuda:1', type=str, help='device: cuda or cpu')
parser.add_argument('--print_freq', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

config = ['--epochs', '50', '--teachers', 'ResNet50', 'ResNet50', '--T', '1.0', '--device', 'cuda:0']
args = parser.parse_args(config)

device = args.device if torch.cuda.is_available() else 'cpu'
load_dir = './model_path/'

# teachers model
teacher_models = []
for te in range(len(args.teachers)):
    te_model = getattr(models, args.teachers[te])(num_classes=args.n_class)
    #     print(te_model)
    print(load_dir + args.teachers_dir[te] + '/' + args.teachers[te] + '.pth')
    te_model.load_state_dict(torch.load(load_dir + args.teachers_dir[te] + '/' + args.teachers[te] + '.pth'))


    te_model.to(device)
    for name, parmas in te_model.named_parameters():
        # if 'linear' in name or 'fc' in name:
        parmas.requires_grad = True
        # else:
        #     parmas.requires_grad = False
    teacher_models.append(te_model)

st_model = getattr(models, args.student)(num_classes=args.n_class)  # args.student()
st_model.load_state_dict(torch.load('source_models_path'))
st_model.to(device)

# logging
logfile = load_dir + 'source_model_name_' + st_model.model_name + '.log'
if os.path.exists(logfile):
    os.remove(logfile)


def log_out(info):
    f = open(logfile, mode='a')
    f.write(info)
    f.write('\n')
    f.close()
    print(info)


def image_train(resize_size=128, crop_size=112):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def image_test(resize_size=128, crop_size=112):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    predict = torch.sigmoid_(output)
    AUROCs = []
    gt_np = target.cpu().detach().numpy()
    pred_np = predict.cpu().detach().numpy()
    # print(pred_np)
    # print(gt_np)
    # gt_np_one = gt_np + np.ones(gt_np.shape,dtype=np.int)
    # pred_np_one = pred_np + 1
    # print(gt_np_one)

    row, col = gt_np.shape
    # print(row)
    # print(col)

    for i in range(col):
        # print(gt_np[:, i])
        fpr, tpr, thresholds = metrics.roc_curve(gt_np[:, i], pred_np[:, i], pos_label=1)
        s = metrics.auc(fpr, tpr)
        AUROCs.append(s)
    # print(AUROCs)
    AUROC_avg = np.array(AUROCs).mean()
    # print(AUROC_avg)
    return AUROC_avg
# adapter model
# adapter model

def update_ema_variables(model, ema_model,alpha):
    for te_model in ema_model:
        for ema_param, param in zip(te_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
class Adapter():
    def __init__(self, in_models, pool_size):
        # representations of teachers
        pool_ch = pool_size[1]  # 64
        pool_w = pool_size[2]  # 8
        LR_list = []
        torch.manual_seed(1)
        self.theta = torch.randn(len(in_models), pool_ch).to(device)  # [3, 64]
        self.theta.requires_grad_(True)

        self.max_feat = nn.MaxPool2d(kernel_size=(pool_w, pool_w), stride=pool_w).to(device)
        self.W = torch.randn(pool_ch, 6).to(device)
        self.W.requires_grad_(True)
        self.val = False

    def loss(self, y, labels, weighted_logits, T=10.0, alpha=0.7):
        # print(F.softmax(y))
        # print(weighted_logits)

        clone_y = y.clone().detach()
        clone_weighted_logits = weighted_logits.clone().detach()

        F.normalize(clone_y, p=1, dim=1)
        F.normalize(clone_weighted_logits, p=1, dim=1)

        # weighted_logits_normed = weighted_logits / weighted_logits.max(axis=0)
        # ls = nn.KLDivLoss()(F.log_softmax(clone_y), clone_weighted_logits) * (T * T * 2.0 * alpha)  + F.binary_cross_entropy(torch.sigmoid(y),
        #                                                                                                      labels.float()) * (
        #                  1. - alpha)
        l1 = nn.KLDivLoss()(F.log_softmax(clone_y / T), clone_weighted_logits) * (T * T * 2.0 * alpha)
        l2 = F.binary_cross_entropy(torch.sigmoid(y),labels.float()) * ( 1. - alpha)
        ls = l1 + l2
        # print("l1={}, l2={}".format(l1, l2))
        # if not self.val:
        # ls += 0.1 * (torch.sum(self.W * self.W) + torch.sum(torch.sum(self.theta * self.theta, dim=1), dim=0))
        return ls

    def gradient(self, lr=0.001):
        self.W.data = self.W.data - lr * self.W.grad.data
        # Manually zero the gradients after updating weights
        self.W.grad.data.zero_()

    def eval(self):
        self.val = True
        self.theta.detach()
        self.W.detach()

    # input size: [64, 8, 8], [128, 3, 10]
    def forward(self, conv_map, te_logits_list):
        # print(conv_map.size()) #torch.Size([4, 2048, 3, 3])
        # print(te_logits_list.size()) #torch.Size([4, 2, 6])

        beta = self.max_feat(conv_map)
        beta = torch.squeeze(beta)  # [128, 64]
        # print(beta.size()) #torch.Size([4, 2048])


        latent_factor = []
        for t in self.theta:
            # print(t.size()) #[2048]

            latent_factor.append(beta * t)  # [4,2048] * [2048]
        #         latent_factor = torch.stack(latent_factor, dim=0)  # [3, 128, 64]
        #print(latent_factor)  2 个 [4, 2048]

        alpha = []
        for lf in latent_factor:  # lf.size:[128, 64]
            alpha.append(lf.mm(self.W)) #[4,2048] * [2048, 6] = [4, 6]
        # alpha 2 个 [4，6]
        # print(alpha[1])
        alpha = torch.stack(alpha, dim=0)  # [3, 128, 1]
        alpha = torch.squeeze(alpha).transpose(0, 1)  # [128, 3]


        miu = F.softmax(alpha, dim=1)  # [128, 3]
        # print(miu)
        weighted_logits = miu.mul(te_logits_list)  
        # print(weighted_logits)


        weighted_logits = torch.sum(weighted_logits, dim=1)
        # print(weighted_logits)[4,6]
        # sys.exit()
        return weighted_logits


# adapter instance
_, _, _, pool_m, _ = st_model(torch.randn(8, 3, 112, 112).to(device))  # get pool_size of student
# reate adapter instance
adapter = Adapter(teacher_models, pool_m.size())

# data
dsets = {}
txt_train_l = open('train_labeled.txt').readlines()
txt_train_u = open('train_unlabeled.txt').readlines()

txt_test = open('test.txt').readlines()
dsets["train_l"] = ImageList_idx(txt_train_l, transform=image_train())
dsets["train_u"] = ImageList_idx(txt_train_u, transform=image_train())
dsets["test"] = ImageList_idx(txt_test, transform=image_test())
train_loader_label = DataLoader(dsets["train_l"], batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True,drop_last=True)
train_loader_unlabel = DataLoader(dsets["train_u"], batch_size=args.batch_size, shuffle=True, num_workers=4,
                               pin_memory=True,drop_last=True)
test_loader = DataLoader(dsets["test"], batch_size=args.test_batch_size, shuffle=False, num_workers=4,
                              pin_memory=True,drop_last=True)

# optim
optimizer_W = optim.SGD([adapter.W], lr=args.lr, momentum=0.9)
optimizer_theta = optim.SGD([adapter.theta], lr=args.lr, momentum=0.9)
optimizer_sgd = optim.SGD(st_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sgd, gamma=0.1, milestones=[30, 50])
lr_scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer_W, milestones=[30, 45])
lr_scheduler3 = optim.lr_scheduler.MultiStepLR(optimizer_theta, milestones=[30, 45])

optimizer_t_sgd = []
lr_scheduler_teacher = []
for te in range(len(args.teachers)):
    optimizer_t = optim.SGD(teacher_models[te].linear.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    lr_scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=[5, 10])
    optimizer_t_sgd.append(optimizer_t)
    lr_scheduler_teacher.append(lr_scheduler_t)
#
# torch.autograd.set_detect_anomaly(True)
# losses
dist_criterion = RkdDistance().to(device)
angle_criterion = RKdAngle().to(device)
# triplet loss
triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2).to(device)

#coordinating weight learning
def train_adapter(n_epochs=70, model=st_model):
    print('Training adapter:')
    start_time = time.time()
    model.train()
    teacher_models[0].eval()
    teacher_models[1].eval()
    # test(st_model)

    for ep in range(n_epochs):
        lr_scheduler2.step()
        lr_scheduler3.step()
        for i, (input, target, idx) in enumerate(train_loader_label):
            # if (i % 10000 == 0):
            #     print(i*10000)
            # print(i)

            input, target = input.to(device), target.to(device)
            # compute outputs
            b1, b2, b3, pool, output = model(input)  # out_feat: 16, 32, 64, 64, -
            # print('b1:{}, b2:{}, b3{}, pool:{}'.format(b1.size(), b2.size(), b3.size(), pool.size())) #b1:torch.Size([16, 8, 112, 112]), b2:torch.Size([16, 16, 56, 56]), b3torch.Size([16, 32, 28, 28]), pool:torch.Size([16, 32, 1, 1])
            st_maps = [b1, b2, b3, pool]
            #             print('b1:{}, b2:{}, b3{}, pool:{}'.format(b1.size(), b2.size(), b3.size(), pool.size()))

            te_scores_list = []
            hint_maps = []
            for j, te in enumerate(teacher_models):
                #                 te.eval()
                with torch.no_grad():
                    t_b1, t_b2, t_b3, t_pool, t_output = te(input)

                #                 print('t_b1:{}, t_b2:{}, t_b3{}, t_pool:{}'.format(t_b1.size(), t_b2.size(), t_b3.size(), t_pool.size()))
                hint_maps.append([t_b1, t_b2, t_b3, t_pool])
                t_output = F.softmax(t_output / args.T)
                # t_output = F.sigmoid(t_output / args.T)
                # t_output = F.sigmoid(t_output)
                # t_output = F.sigmoid(t_output / args.T)
                te_scores_list.append(t_output)
            te_scores_Tensor = torch.stack(te_scores_list, dim=1)  # size: [128, 3, 10]

            optimizer_sgd.zero_grad()
            optimizer_W.zero_grad()
            optimizer_theta.zero_grad()


            # st_tripets = random_triplets(b2, t_b2)
            # relation_loss = triplet_loss(st_tripets[0], st_tripets[1], st_tripets[2])

            weighted_logits = adapter.forward(pool, te_scores_Tensor)
            # print(weighted_logits)

            #angle_loss = angle_criterion(output, weighted_logits)
            #dist_loss = dist_criterion(output, weighted_logits)
            # compute gradient and do SGD step
            ada_loss = adapter.loss(output, target, weighted_logits, T=args.T, alpha=args.kd_ratio)

            # loss = ada_loss + angle_loss + dist_loss
            loss = ada_loss
            # loss = ada_loss
            # print("l1={}, l2={}, angle_loss={}, dist_loss={}, relations_loss={}".format(l1, l2, angle_loss, dist_loss, relation_loss))

            loss.backward(retain_graph=True)
            optimizer_sgd.step()
            optimizer_W.step()
            optimizer_theta.step()

        #          vis.line(np.array([loss.item()]), np.array([ep]), loss_win, update="append")
        log_out('epoch[{}/{}]adapter Loss: {:.4f}'.format(ep, n_epochs, loss.item()))
        log_out('adapter.theta={},adapter.W={}'.format(adapter.theta, adapter.W))
        # print(adapter.theta * adapter.W)
        np_theta = adapter.theta.detach().cpu().numpy()
        np_W = adapter.W.detach().cpu().numpy()


        test_result = test(st_model)
        test_acc_s = test_result[1]
    end_time = time.time()
    log_out("--- adapter training cost {:.3f} mins ---".format((end_time - start_time) / 60))



# bilevel_optimization

def train(epoch, model, globals_step):
    if epoch > 5:
        train_adapter(1, model)
    print('Training:')
    # switch to train mode
    model.train()
    t_max_feat = nn.MaxPool2d(kernel_size=(pool_m.size()[2], pool_m.size()[2]), stride=pool_m.size()[2]).to(device)

    for te in range(len(args.teachers)):
        teacher_models[te].train()
    adapter.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()
    labeled_train_iter = iter(train_loader_label)
    end = time.time()
    for i, (input, target, idx) in enumerate(train_loader_unlabel):
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        b1, b2, b3, pool, output = model(input)

        te_scores_list = []
        te_pools_list = []
        for j, te in enumerate(teacher_models):
            te.eval()
            with torch.no_grad():
                t_b1, t_b2, t_b3, t_pool, t_output = te(input)
            t_output_s = F.sigmoid(t_output / args.T)
            te_scores_list.append(t_output_s)
            te_pools_list.append(t_pool)

        te_scores_Tensor = torch.stack(te_scores_list, dim=1)  # size: [128, 3, 10]
        weighted_logits = adapter.forward(pool, te_scores_Tensor).detach()
        optimizer_sgd.zero_grad()
        # angle_loss = angle_criterion(output, weighted_logits)
        # dist_loss = dist_criterion(output, weighted_logits)
        y_hat1 = weighted_logits.detach()
        zero = torch.zeros_like(y_hat1)
        one = torch.ones_like(y_hat1)
        y_hat1 = torch.where(y_hat1 >= 0.5, one, y_hat1)
        y_hat1 = torch.where(y_hat1 < 0.5, zero, y_hat1)
        y_hat1.detach()

        
        l1 = (nn.BCELoss()(nn.Sigmoid()(output / args.T), y_hat1))
        l1.backward(retain_graph=True)
        with torch.no_grad():
            grad_s_on_u = []
            for name, params in model.named_parameters():
                # print(name)
                # if 'linear' in name or 'fc' in name:
                grad_s_on_u.append(params.grad.view(-1))
            grad_s_on_u = torch.cat(grad_s_on_u)

        optimizer_sgd.step()  ## now new student updated.
        # optimizer_sgd.zero_grad()
        # #
        # #
        # b1, b2, b3, pool, output = model(input)
        # optimizer_sgd.zero_grad()

        ########################################################
        ##################################################
        ##################################
        # b1, b2, b3, pool, output = model(input)
        try:
            inputs_l, target_l, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(train_loader_label)
            inputs_l, target_l, _ = labeled_train_iter.next()
       
        inputs_l, target_l = inputs_l.to(device), target_l.to(device)
        _,_,_,_,s_out_on_l = model(inputs_l)
        loss_s_on_l = nn.BCELoss()(F.sigmoid(s_out_on_l), target_l.float().detach())
        optimizer_sgd.zero_grad()
        loss_s_on_l.backward()
        
        with torch.no_grad():
            grad_s_on_l = []
            for name, params in model.named_parameters():
                # if 'linear' in name or 'fc' in name:
                grad_s_on_l.append(params.grad.view(-1))
            grad_s_on_l = torch.cat(grad_s_on_l)

        with torch.no_grad():
            dot_product = grad_s_on_u * grad_s_on_l
            dot_product = dot_product.detach()

       



        # l2 = (nn.BCELoss()(nn.Sigmoid()(output / args.T), target.float())) * (1. - args.kd_ratio)  
        # loss_s_on_l = l2
        # loss_s_on_l.backward(retain_graph=True)
        # with torch.no_grad():
        #     grad_s_on_l = []
        #     for name, params in model.named_parameters():
        #         if 'linear' in name or 'fc' in name:
        #             grad_s_on_l.append(params.grad.view(-1))
        #     grad_s_on_l = torch.cat(grad_s_on_l)

        ############################
        ##############################################
        #####################################################
        # optimizer_sgd.step()
        # optimizer_sgd.zero_grad()## update student weights on reals
        # ada_loss = l2 + l1
        # loss = ada_loss


        #teacher on u
        for te in range(len(args.teachers)):
            # print(te)
            # optimizer_t = optim.SGD(teacher_models[te].linear.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
            # lr_scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=[5, 10])
            # optimizer_t_sgd.append(optimizer_t)
            # lr_scheduler_teacher.append(lr_scheduler_t)
            optimizer_t_sgd[te].zero_grad()
            _, _, _, te_pools_1, te_output = teacher_models[te](input)
            loss_t_on_u = nn.BCELoss()(F.sigmoid(te_output / args.T), y_hat1.detach())
            loss_t_on_u.backward(retain_graph=True)
            with torch.no_grad():
                grad_t_on_u = []
                for name, params in teacher_models[te].named_parameters():
                    # if 'linear' in name or 'fc' in name:
                    grad_t_on_u.append(params.grad.view(-1))
                grad_t_on_u = torch.cat(grad_t_on_u)

                grad_t_on_u = grad_t_on_u * dot_product  # Total Grad for teacher

            
            _,_,_,_,te_output_on_l = teacher_models[te](inputs_l)
            loss_t_on_l = F.binary_cross_entropy(F.sigmoid(te_output_on_l/args.T), target_l.float())

            loss_between_teachers = 0.0
            for i in range(len(args.teachers)):
                if i != te:
                    te_pools_2 = te_pools_list[i].clone().detach()
                    t_max_feat(te_pools_1)
                    te_pools_minus = t_max_feat(te_pools_2) - t_max_feat(te_pools_1)
                    loss_between_teachers += torch.log(torch.norm(te_pools_minus, p=2))
            # print("loss_between_teacher = {}, loss_t_on_l = {}".format(loss_between_teachers, loss_t_on_l))
            loss_t_on_l -= loss_between_teachers

            optimizer_t_sgd[te].zero_grad()
            loss_t_on_l.backward()
                # optimizer_t[te].zero_grad()
                # loss_t_on_l.backward()

                #add meta grad
                # add Meta grad
            for name, params in teacher_models[te].named_parameters():
                # if 'linear' in name or 'fc' in name:
                grad_size = params.grad.view(-1).size(0)
                grad_shape = params.grad.shape
                meta_grad = grad_t_on_u[:grad_size]
                meta_grad = meta_grad.reshape(grad_shape)
                params.grad += meta_grad

                grad_t_on_u = grad_t_on_u[grad_size:]

            optimizer_t_sgd[te].step()
        ###############################

        # loss = ada_loss

        # print("l1={}".format(l1))
        # print("l1={}, l2={}".format(l1, l2))
        # print("ada_loss={}, angle_loss={}, dist_loss={}, relations_loss={}".format(ada_loss, angle_loss, dist_loss,
        #                                                                            relation_loss))

        # loss.backward(retain_graph=True)


        ####################mean teacher##################
        # alpha = 0.95
        # update_ema_variables(model, teacher_models, alpha)
        # globals_step += 1
        ################################################

        output = output.float()
        loss = l1
        loss = loss.float()
        losses.update(loss.item(), input.size(0))
        # measure accuracy and record loss
        #               train_acc = accuracy(output.data, target.data)[0]
        # losses.update(loss.item(), input.size(0))
        # top1.update(train_acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_out('[{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(train_loader_unlabel), batch_time=batch_time,
                data_time=data_time, loss=losses))
    return losses.avg, globals_step


def test(model):
    print('Testing:')
    # switch to evaluate mode
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    output_list_np = np.zeros((1, 6), dtype=float)
    target_list_np = np.zeros((1, 6), dtype=float)


    with torch.no_grad():
        for i, (input, target, idx) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            _, _, _, _, output = model(input)
            #            loss = F.cross_entropy(output, target)
            loss = F.binary_cross_entropy_with_logits(output, target.float())

            output = output.float()
            loss = loss.float()

            output_list_np = np.concatenate((output_list_np, output.cpu().detach().numpy()), axis=0)
            target_list_np = np.concatenate((target_list_np, target.cpu().detach().numpy()), axis=0)

            # measure accuracy and record loss
            # test_acc = accuracy(output.data, target.data)[0]
            losses.update(loss.item(), input.size(0))
            # top1.update(test_acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log_out('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses))
    output_list_np = np.delete(output_list_np, 0, axis=0)
    target_list_np = np.delete(target_list_np, 0, axis=0)
    test_acc = accuracy(torch.from_numpy(output_list_np), torch.from_numpy(target_list_np))
    log_out(' * Prec@1 {test_acc:.3f}'.format(test_acc=test_acc))

    return losses.avg, test_acc



print('StudentNet:\n')
print(st_model)
# test(st_model)
st_model.apply(weights_init_normal)

st_model.load_state_dict(torch.load('random_teacher_model'))
train_adapter(n_epochs=4, model=st_model)



best_acc = 0
globals_step = 0
for epoch in range(1, args.epochs + 1):
    log_out("\n===> epoch: {}/{}".format(epoch, args.epochs))
    log_out('current lr {:.5e}'.format(optimizer_sgd.param_groups[0]['lr']))
    lr_scheduler.step(epoch)

    train_loss, g_s = train(epoch, st_model, globals_step)

    # visaulize loss
    test_result = test(st_model)
    test_acc_s = test_result[1]
    print(test_acc_s)

    test(teacher_models[0])
    test(teacher_models[1])

    if test_acc_s > best_acc:
        best_acc = test_acc_s
        torch.save(st_model.state_dict(), 'save_path' )

# release GPU memory
torch.cuda.empty_cache()
log_out("BEST ACC: {:.3f}".format(best_acc))
log_out("--- {:.3f} mins ---".format((time.time() - start_time) / 60))
# """
