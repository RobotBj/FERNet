'''
Dual Refinement Underwater Object Detection Network
'''
from __future__ import print_function
import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from models.FERrefineVgg import build_net
from layers.modules import RefineMultiBoxLoss
from data import VOCroot, COCOroot, VOC_300, COCO_300, URPC_300, AnnotationTransform, COCODetection, URPCAnnotationTransform,\
    VOCDetection, detection_collate, BaseTransform, preproc, URPCDetection, URPCroot


parser = argparse.ArgumentParser(description='Feature Enhancement and Refinement Network (FERNet)')
parser.add_argument('-v', '--version', default='FER_vgg', help='FER_vgg  or FER_mobile version.')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='URPC', help='URPC or VOC or COCO dataset')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
# parser.add_argument('--basenet', default='./weights/vgg16bn_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=6, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=160, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = VOC_300
elif args.dataset == 'URPC':
    train_sets = None
    cfg = URPC_300
else:
    train_sets = [('2014', 'train'), ('2014', 'valminusminival')]
    cfg = COCO_300


seed = 888888
np.random.seed(seed)
torch.manual_seed(seed)


img_dim = (300, 512)[args.size == '512']
if args.dataset == 'VOC':
    rgb_means = (103.94, 116.78, 123.68)
elif args.dataset == 'COCO':
    rgb_means = (104, 117, 123)
else:
    rgb_means = (64.238223, 142.588190, 56.799253)

p = (0.6, 0.2)[args.version == 'FER_mobile']

if args.dataset == 'COCO':
    num_classes = 81
elif args.dataset == 'VOC':
    num_classes = 21
else:
    num_classes = 5


batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9


net = build_net('train', cfg, img_dim, num_classes)
print(net)

if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
# initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.arm_loc.apply(weights_init)
    net.arm_conf.apply(weights_init)
    net.odm_loc.apply(weights_init)
    net.odm_conf.apply(weights_init)
    net.Norm.apply(weights_init)
    net.loc_offset_conv.apply(weights_init)
    net.dcn_convs.apply(weights_init)
else:
# load resume network
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
# create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = list()
arm_criterion = RefineMultiBoxLoss(2)
odm_criterion = RefineMultiBoxLoss(num_classes)
criterion.append(arm_criterion)
criterion.append(odm_criterion)



def train():
    net.train()
# loss counters
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    elif args.dataset == 'URPC':
        dataset = URPCDetection('train', URPCroot,  preproc(
            img_dim, rgb_means, p), URPCAnnotationTransform())
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))

            if (epoch % 20 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(),
                           args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        epoch_step = [90, 110, 130, 150, 160]

        lr = adjust_learning_rate(optimizer, args.gamma, epoch, epoch_step, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        arm_criterion = criterion[0]
        odm_criterion = criterion[1]
        arm_loss_l, arm_loss_c = arm_criterion(out,  targets)
        odm_loss_l, odm_loss_c = odm_criterion(out,  targets, use_arm=True, filter_object=True)
        loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        load_t1 = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(
                epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || arm_L: %.4f arm_C: %.4f||' %
                  (arm_loss_l.item(), arm_loss_c.item()) +
                  ' odm_L: %.4f odm_C: %.4f||' %
                  (odm_loss_l.item(), odm_loss_c.item()) +
                  ' loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))

    torch.save(net.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '.pth')



def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    LR = [0.004, 0.002, 0.0004, 0.00004, 0.000004]
    if epoch <= 6:
        lr = 1e-6 + (LR[0] - 1e-6) * iteration / (epoch_size * 5)
    else:
        for i in range(len(step_index)):
            if step_index[i] >= epoch:
                lr = LR[i]
                break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




if __name__ == '__main__':
    train()