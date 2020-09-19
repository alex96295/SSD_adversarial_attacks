import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# from darknet import Darknet

from median_pool import MedianPool2d  # see median_pool.py

# print('Test image loading on a random image:')
# im = Image.open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/horse.jpg').convert('RGB')
# print('Image has been read correctly!')


class yolov2_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov2_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput, loss_type):
        # get values necessary for transformation

        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)  # add one dimension of size 1
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls ) * 5)  # the last 5 is the anchor boxes number after k-means clustering
                                                                # the first 5 is the number of parameters of each box: x, y, w, h, objectness score
                                                                # self.num_cls indicates class probabilities, i.e. 20 values for VOC and 80 for COCO
                                                                # in total, there are 125 parameters per grid cell when VOC, 425 when COCO
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)

        # print('dim0:' + str(YOLOoutput.size(0)))
        # print('dim1:' + str(YOLOoutput.size(1)))
        # print('h:' + str(h))
        # print('w:' + str(h))


        # transform the output tensor from [batch, 425, 13, 13] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls , h * w)  # [batch, 5, 85, 169]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 169] swap 5 and 85, in position 1 and 2 respectively
        output = output.view(batch, 5 + self.num_cls , 5 * h * w)  # [batch, 85, 845]
        # todo first 5 numbers that make '85' are box xc, yc, w, h and objectness. Last 80 are class prob.

        # print(output[:, 4, :])
        # print(output[:, 5, :])
        # print(output[:, 6, :])
        # print(output[:, 34, :])

        output_objectness_not_norm = output[:, 4, :]
        output_objectness_norm = torch.sigmoid(output[:, 4, :])  # [batch, 1, 845]  # iou_truth*P(obj)
        # take the fifth value, i.e. object confidence score. There is one value for each box, in total 5 boxes

        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 845]  # 845 = 5 * h * w
        # NB 80 means conditional class probabilities, one for each class related to a single box (there are 5 box for each grid cell)

        # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
        not_normal_confs = output
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1

        # we only care for conditional probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
        confs_for_class_not_normal = not_normal_confs[:, self.cls_id, :]
        confs_for_class_normal = normal_confs[:, self.cls_id, :] # take class number 0, so just one kind of cond. prob out of 80. This is for 1 box, there are 5 boxes

        confs_if_object_not_normal = self.config.loss_target(output_objectness_not_norm, confs_for_class_not_normal)
        confs_if_object_normal = self.config.loss_target(output_objectness_norm, confs_for_class_normal)  # loss_target in patch_config

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('yolo batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf


class ssd_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(ssd_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, ssd_output, loss_type):
        # get values necessary for transformation
        conf_normal, conf_not_normal, loc = ssd_output

        # print(conf.size())
        # print(loc.size())
        # print(conf[:,:,self.cls_id])
        # max, max_id = torch.max(conf[:,:,self.cls_id], dim=1)
        # print(max)

        confs_if_object_not_normal = conf_not_normal[:, :, self.cls_id] # no softmax yet
        confs_if_object_normal = conf_normal[:, :, self.cls_id] # softmaxed

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1) # take the maximum value among your 5 priors
            return max_conf
        elif loss_type == 'threshold_approach':
            threshold = 0.35
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            #print('ssd batch stack: \n')
            #print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.cuda.FloatTensor(size).fill_(0)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor)**2
                #penalized_tensor = penalized_tensor**2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf

        # code below good for coco training
        # loc_data = loc.data
        # conf_data = conf.data
        # print(conf_data.size())
        # num_priors = self.num_priors.data.size(0)
        #
        # batch = loc_data.size(0)
        #
        # output = conf_data.view(batch, num_priors, self.num_cls)
        # output = output[:, :, self.cls_id]
        # max_conf, max_conf_idx = torch.max(output, dim=1)


class yolov3_feature_output_manage(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(yolov3_feature_output_manage, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        #self.num_priors = num_priors

    def forward(self, yv3_output, loss_type):
        # get values necessary for transformation

        yolo_output = yv3_output[0]

        loc = yolo_output[:, :, :4]
        objectness = yolo_output[:, :, 4]
        cond_prob = yolo_output[:, :, 5:]

        cond_prob_targeted_class = cond_prob[:, :, self.cls_id]

        confs_if_object_normal = self.config.loss_target(objectness, cond_prob_targeted_class)

        if loss_type == 'max_approach':
            max_conf, max_conf_idx = torch.max(confs_if_object_normal, dim=1)
            return max_conf

        elif loss_type == 'threshold_approach':
            threshold = 0.3
            batch_stack = torch.unbind(confs_if_object_normal, dim=0)
            print('ssd batch stack: \n')
            print(batch_stack)
            penalized_tensor_batch = []
            for img_tensor in batch_stack:
                size = img_tensor.size()
                zero_tensor = torch.zeros(size)
                penalized_tensor = torch.max(img_tensor - threshold, zero_tensor) ** 2
                penalized_tensor_batch.append(penalized_tensor)

            penalized_tensor_batch = torch.stack(penalized_tensor_batch, dim=0)
            thresholded_conf = torch.sum(penalized_tensor_batch, dim=1)
            return thresholded_conf

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference

        color_dist = (adv_patch - self.printability_array+0.000001)
        #print(color_dist.size())
        color_dist = color_dist ** 2  # squared difference
        color_dist = torch.sum(color_dist, 1)+0.000001
        #print(color_dist.size())
        color_dist = torch.sqrt(color_dist)

        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        #print(type(color_dist_prod))
        #print('size ' + str(color_dist_prod.size()))

        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)  # divide by the total number of elements in the input tensor

    def get_printability_array(self, printability_file, side):
        #  side = patch_size in adv_examples.py
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        # see notes for a better graphical representation
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))

            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)  # convert input lists, tuples etc. to array
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)  # Creates a Tensor from a numpy array.
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # compute total variation of the adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)

        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):

        use_cuda=0
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))

        # adv_batch_im = transforms.ToPILImage('RGB')(adv_patch[0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)

        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        if use_cuda:
            contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        else:
            contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)


        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast

        # Create random brightness tensor
        if use_cuda:
            brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        else:
            brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)

        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

        # Create random noise tensor
        if use_cuda:
            noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        else:
            noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # adv_batch_im = transforms.ToPILImage('RGB')(adv_batch[0][0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))

        if use_cuda:
            msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
        else:
            msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        msk_batch_im = transforms.ToPILImage('RGB')(msk_batch[0][0])
        # plt.imshow(msk_batch_im)
        # plt.show()

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            if use_cuda:
                angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
            else:
                angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            if use_cuda:
                angle = torch.FloatTensor(anglesize).fill_(0)
            else:
                angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        if use_cuda:
            lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        else:
            lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0)

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if (rand_loc):
            if use_cuda:
                off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            else:
                off_x = targetoff_x * (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))

            target_x = target_x + off_x

            if use_cuda:
                off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            else:
                off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))

            target_y = target_y + off_y

        target_y = target_y - 0.05
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        if use_cuda:
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        else:
            theta = torch.FloatTensor(anglesize, 2, 3).fill_(0)

        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)

        print(adv_batch_t.size())

        # adv_batch_im = transforms.ToPILImage('RGB')(adv_batch_t[0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        msk_batch_t = F.grid_sample(msk_batch, grid)

        # msk_batch_im = transforms.ToPILImage('RGB')(msk_batch_t[0])
        # plt.imshow(msk_batch_im)
        # plt.show()

        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_im = transforms.ToPILImage('RGB')(adv_batch_t[0][0])
        # plt.imshow(adv_batch_im)
        # plt.show()

        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        msk_batch_im = transforms.ToPILImage('RGB')(msk_batch_t[0][0])
        # plt.imshow(msk_batch_im)
        # plt.show()

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        #prod_im = transforms.ToPILImage('RGB')(adv_batch_t[0][0]*msk_batch_t[0][0])
        # plt.imshow(prod_im)
        # plt.show()

        return adv_batch_t * msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            print(adv)
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


#TODO ____________________________________________________________________________________________________________________________________________________________
    # TODO Summary of PatchTransformer + PatchApplier:
    # take a batch of 6 images, consider one. For it, I have 14 ready adv patches, of which a number that varies for each image is non-zero (remember:
    # the mask is done starting from 0 and 1 labels in lab_batch. Suppose that 5 are non zero. It means that they correspond to 5 detected object in that image.
    # They are already transformed according to correct positions and scales of the 5 detected rectangles. Now, we consider the image of the six composing the batch,
    # and substitute the patches in their positions where they are not zero (so 5 out of 14 in this example)
#TODO ____________________________________________________________________________________________________________________________________________________________


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        #imgsize = 416 from yolo
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # to make it agrees with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        return image, label

    def pad_and_scale(self, img, lab): # this method for taking a non-square image and make it square by filling the difference in w and h with gray
                                       # needed to keep proportions
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize)) # make a square image of dim 416 x 416
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                   # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                   # whose number is not known a priori
        else:
            padded_lab = lab
        return padded_lab
