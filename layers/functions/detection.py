import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
from utils.box_utils import decode, center_size


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.object_score = 0.01
        self.variance = cfg['variance']

    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        # start_time = time.time()
        arm_loc, arm_conf, loc, conf, priors = predictions
        arm_conf = F.softmax(arm_conf.view(-1, 2), 1)
        conf = F.softmax(conf.view(-1, self.num_classes), 1)
        arm_loc_data = arm_loc.data
        arm_conf_data = arm_conf.data
        arm_object_conf = arm_conf_data[:, 1:]
        no_object_index = arm_object_conf <= self.object_score
        conf.data[no_object_index.expand_as(conf.data)] = 0
        # time1 = time.time() - start_time
        # print('prediction_time_first:', time1)


        # start_time2 = time.time()
        loc_data = loc.data
        conf_data = conf.data
        # prior_data = priors.data
        prior_data = priors[:loc_data.size(1),:]

        num = loc_data.size(0)  # batch size

        self.num_priors = prior_data.size(0)
        # time2 = time.time() - start_time2
        # print('prepare_time:', time2)

        # start_time3 = time.time()
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)
        conf_preds = conf_data.view(num, self.num_priors, self.num_classes)
        batch_prior = prior_data.view(-1, self.num_priors, 4).expand(
            (num, self.num_priors, 4))
        batch_prior = batch_prior.contiguous().view(-1, 4)
        # time3 = time.time() - start_time3
        # print('prepare_time2:', time3)

        # start_time4 = time.time()

        default = decode(
            arm_loc_data.view(-1, 4), batch_prior, self.variance)
        default = center_size(default)
        decoded_boxes = decode(
            loc_data.view(-1, 4), default, self.variance)


        self.scores = conf_preds.view(num, self.num_priors, self.num_classes)
        self.boxes = decoded_boxes.view(num, self.num_priors, 4)

        # time4 = time.time() - start_time4
        # print('prediction_time2:', time4)
        return self.boxes, self.scores