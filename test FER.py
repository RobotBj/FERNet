'''
Dual Refinement Underwater Object Detection Network
'''
from __future__ import print_function
import os
import sys
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from utils.timer import Timer
import torch.utils.data as data
from utils.nms_wrapper import nms
from data import VOCroot, COCOroot, URPCroot
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from models.FERrefineVgg import build_net
import torchvision.transforms as transforms
from layers.functions import Detect
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300, VOC_512, COCO_300, COCO_512,\
    URPCDetection, URPC_300

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


parser = argparse.ArgumentParser(description='Feature Enhancement and Refinement Network (FERNet)')
parser.add_argument('-v', '--version', default='FER_vgg', help='FER_vgg ,FER_E_vgg or FER_mobile version.')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool, help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
if args.dataset == 'COCO':
    cfg = (COCO_300, COCO_512)[args.size == '512']
else:
    cfg = URPC_300



def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    if args.dataset == 'VOC'or'COCO':
        num_classes = (21, 81)[args.dataset == 'COCO']
    if args.dataset == 'URPC':
        num_classes = 5

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')

        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')

    if args.dataset != 'URPC':
        testset.evaluate_detections(all_boxes, save_folder)
    else:
        testset.evaluate_URPC_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == '512']
    if args.dataset == 'VOC' or 'COCO':
        num_classes = (21, 81)[args.dataset == 'COCO']
    if args.dataset == 'URPC':
        num_classes = 5

    net = build_net('test', cfg, img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)

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
    net.eval()
    print('Finished loading model!')
    print(net)

    # load data
    if args.dataset == 'VOC':
        testset = VOCDetection(VOCroot, [('2007', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(COCOroot, [('2014', 'minival')], None)
    elif args.dataset == 'URPC':
        testset = URPCDetection('test', URPCroot, None, None, AnnotationTransform())
    else:
        print('Only UPRC2020, VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    # evaluation
    top_k = 200
    detector = Detect(num_classes, 0, cfg)
    save_folder = os.path.join(args.save_folder, args.dataset)
    if args.dataset == 'VOC'or'COCO':
        rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'FER_mobile']
    else:
        rgb_means = (64.238223, 142.588190, 56.799253)

    test_net(save_folder, net, detector, args.cuda, testset,
             BaseTransform(img_dim, rgb_means, (2, 0, 1)),
             top_k, thresh=0.001)
