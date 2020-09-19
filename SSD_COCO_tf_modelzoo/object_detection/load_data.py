import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
from PIL import Image
import stn
import torch

# pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

# keras
import tensorflow.keras as tfk
import tensorflow as tf

#from darknet import Darknet

#from median_pool import MedianPool2d  # see median_pool.py

# print('Test image loading on a random image:')
# im = Image.open('C:/Users/Alessandro/PycharmProjects/YOLOv2_d2p_adv_COCO_thys2019/data/horse.jpg').convert('RGB')
# print('Image has been read correctly!')

class PatchTransformer():
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        #super(PatchTransformer, self).__init__()

        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        #self.medianpooler = MedianPool2d(7, same=True)  # kernel_size = 7? see again
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''
    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):

        use_cuda = 0

        #adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        # adv_patch = self.medianpooler(tfk.backend.expand_dims(adv_patch, 0))  # pre-processing on the image with 1 more dimension: 1 x 3 x 300 x 300, see median_pool.py

        # Determine size of padding
        pad = (img_size - tfk.backend.int_shape(adv_patch)[-1]) / 2  # img_size = 416, adv_patch size = patch_size in adv_examples.py, = 300
        # print('pad =' + str(pad)) # pad = 0.5*(416 - 300) = 58

        # Make a batch of patches
        adv_patch = tfk.backend.expand_dims(adv_patch, 0)
        adv_patch = tfk.backend.expand_dims(adv_patch, 0)
        # print('adv_patch in load_data.py, PatchTransforme, size =' + str(adv_patch.size()))
        # adv_patch in load_data.py, PatchTransforme, size =torch.Size([1, 1, 3, 300, 300]), tot 5 dimensions

        #adv_batch = tfk.backend.reshape(adv_patch, shape=(tfk.backend.int_shape(lab_batch)[0], tfk.backend.int_shape(lab_batch)[1], tfk.backend.int_shape(adv_patch)[2], tfk.backend.int_shape(adv_patch)[3], tfk.backend.int_shape(adv_patch)[4]))
        adv_batch = tfk.backend.repeat_elements(adv_patch, tfk.backend.int_shape(lab_batch)[0], 0)
        adv_batch = tfk.backend.repeat_elements(adv_batch, tfk.backend.int_shape(lab_batch)[1], 1)
        # print('adv_batch in load_data.py, PatchTransforme, size =' + str(adv_batch.size()))
        # adv_batch in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        batch_size = tfk.backend.constant(0.5, shape=(tfk.backend.int_shape(lab_batch)[0], tfk.backend.int_shape(lab_batch)[1]))
        batch_size = tfk.backend.int_shape(batch_size)
        # print('batch_size in load_data.py, PatchTransforme, size =' + str(batch_size))
        # batch_size in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = tfk.backend.random_uniform(batch_size, self.min_contrast, self.max_contrast)
        # Fills self tensor (here 6 x 14) with numbers sampled from the continuous uniform distribution: 1/(max_contrast - min_contrast)

        # print('contrast1 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast1 in load_data.py, PatchTransforme, size =torch.Size([6, 14])

        contrast = tfk.backend.expand_dims(contrast, -1)
        contrast = tfk.backend.expand_dims(contrast, -1)
        contrast = tfk.backend.expand_dims(contrast, -1)

        # print('contrast2 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast2 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 1, 1, 1])

        #contrast = tfk.backend.reshape(contrast, shape=(batch_size[0], batch_size[1], tfk.backend.int_shape(adv_batch)[-3],tfk.backend.int_shape(adv_batch)[-2],tfk.backend.int_shape(adv_batch)[-1]))
        contrast = tfk.backend.repeat_elements(contrast, tfk.backend.int_shape(adv_batch)[-3], 2)
        contrast = tfk.backend.repeat_elements(contrast, tfk.backend.int_shape(adv_batch)[-2], 3)
        contrast = tfk.backend.repeat_elements(contrast, tfk.backend.int_shape(adv_batch)[-1], 4)

        # print('contrast3 in load_data.py, PatchTransforme, size =' + str(contrast.size()))
        # contrast3 in load_data.py, PatchTransforme, size =torch.Size([6, 14, 3, 300, 300])

        # lines 206-221 could be replaced by:
        # contrast = torch.FloatTensor(adv_batch).uniform_(self.min_contrast, self.max_contrast)
        # print('contrast4 in load_data.py, PatchTransforme, size =' + str(contrast.size()))

        if use_cuda:
            contrast = contrast.cuda()
        else:
            contrast = contrast
#_________________________________________________________________________________________________________________________________________________
        # Create random brightness tensor
        brightness = tfk.backend.random_uniform(batch_size, self.min_brightness, self.max_brightness)

        brightness = tfk.backend.expand_dims(brightness, -1)
        brightness = tfk.backend.expand_dims(brightness, -1)
        brightness = tfk.backend.expand_dims(brightness, -1)
        #brightness = tfk.backend.reshape(brightness, shape=(batch_size[0], batch_size[1], tfk.backend.int_shape(adv_batch)[-3],tfk.backend.int_shape(adv_batch)[-2],tfk.backend.int_shape(adv_batch)[-1]))
        brightness = tfk.backend.repeat_elements(brightness, tfk.backend.int_shape(adv_batch)[-3], 2)
        brightness = tfk.backend.repeat_elements(brightness, tfk.backend.int_shape(adv_batch)[-2], 3)
        brightness = tfk.backend.repeat_elements(brightness, tfk.backend.int_shape(adv_batch)[-1], 4)

        # lines 227-239 could be replaced by:
        # brightness = torch.FloatTensor(adv_batch).uniform_(self.min_brightness, self.max_brightness)
        # print('brightness in load_data.py, PatchTransforme, size =' + str(brightness.size()))

        if use_cuda:
            brightness = brightness.cuda()
        else:
            brightness = brightness

# _____________________________________________________________________________________________________________________________________________
        # Create random noise tensor
        noise = tfk.backend.random_uniform(tfk.backend.int_shape(adv_batch), -1, 1) * self.noise_factor
        # dim: 6 x 14 x 3 x 300 x 300
#______________________________________________________________________________________________________________________________________________
        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise
        #print(tfk.backend.int_shape(adv_batch))

        # adv_batch_s = tf.unstack(adv_batch, 0)
        # adv_batch_1 = tf.unstack(adv_batch_s[0], 0)
        #
        # sess = tf.Session()
        # tfk.backend.set_session(sess)
        # adv_batch_2 = sess.run(adv_batch_1[0])  # to np array
        # print(adv_batch_2)

        tfk.backend.clip(adv_batch, 0.000001, 0.99999) # keep all elements in the range 0.000001-0.99999 (real numbers since FLoatTensor)


        # dim: 6 x 14 x 3 x 300 x 300

#______________________________________________________________________________________________________________________________________________
        # Where the label class_ids is 1 we don't want a patch (padding) --> fill mask with zero's

        cls_ids = lab_batch[:,:,:1]  # Consider just the first 'column' of lab_batch, where we can
                                     # discriminate between detected person (or 'yes person') and 'no person')
                                     # in this way, sensible data about x, y, w and h of the rectangles are not used for building the mask


        # NB torch.narrow returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length.
        # The returned tensor and input tensor share the same underlying storage.

        cls_mask = tfk.backend.repeat_elements(cls_ids,3,2)
        cls_mask = tfk.backend.expand_dims(cls_mask, -1)
        cls_mask = tfk.backend.repeat_elements(cls_mask,tfk.backend.int_shape(adv_batch)[3],-1)
        cls_mask = tfk.backend.expand_dims(cls_mask, -1)
        cls_mask = tfk.backend.repeat_elements(cls_mask,tfk.backend.int_shape(adv_batch)[4],-1)  # 6 x 14 x 3 x 300 x 300

        msk_batch = tfk.backend.constant(1, shape=(tfk.backend.int_shape(cls_mask))) - cls_mask   # take a matrix of 1s, subtract that of the labels so that
                                                                            # we can have 0s where there is no person detected,
                                                                            # obtained by doing 1-1=0
        #print(tfk.backend.int_shape(msk_batch))
        # NB! Now the mask has 1s 'above', where the labels data are sensible since they represent detected persons, and 0s where there are no detections
        # In this way, multiplying the adv_batch to this mask, built from the lab_batch tensor, allows to target only detected persons and nothing else,
        # i.e. pad with zeros the rest
#_______________________________________________________________________________________________________________________________________________
        # Pad patch and mask to image dimensions with zeros
        padding = tf.constant([[0,0],[0,0],[0,0],[int(pad + 0.5), int(pad)], [int(pad + 0.5), int(pad)]])
        adv_batch = tf.pad(adv_batch, padding, 'CONSTANT', constant_values=0)
        #print(tfk.backend.int_shape(adv_batch))
        msk_batch = tf.pad(msk_batch, padding, 'CONSTANT', constant_values=0)

        # mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0) # left, right, top, bottom
        # adv_batch = mypad(adv_batch)  # dim 6 x 14 x 3 x 416 x 416
        # msk_batch = mypad(msk_batch)  # dim 6 x 14 x 3 x 416 x 416

        # NB you see only zeros when you print it because they are all surrounding the patch to pad it to image dimensions (3 x 416 x 416)

#_______________________________________________________________________________________________________________________________________________
        # Rotation and rescaling transforms
        anglesize = tfk.backend.int_shape(lab_batch)[0] * tfk.backend.int_shape(lab_batch)[1]  # dim = 6*14 = 84
        if do_rotate:
            angle = tfk.backend.random_uniform((1, anglesize), self.minangle, self.maxangle)
        else:
            angle = tfk.backend.constant(0,shape=(1, anglesize))
#_______________________________________________________________________________________________________________________________________________
        # Resizes and rotates

        #sess = tf.Session()

        current_patch_size = tfk.backend.int_shape(adv_patch)[-1]  # 300

        lab_batch_scaled = tfk.backend.constant(0,shape=(tfk.backend.int_shape(lab_batch)))  # dim 6 x 14 x 5

        #tfk.backend.set_session(sess)
        lab_batch = np.array(lab_batch)
        lab_batch_scaled = np.array(lab_batch_scaled)

        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size

        # l0 = lab_batch[:, :, 0] * 0
        # l1 = lab_batch[:, :, 1] * img_size
        # l2 = lab_batch[:, :, 2] * img_size
        # l3 = lab_batch[:, :, 3] * img_size
        # l4 = lab_batch[:, :, 4] * img_size
        #
        # l_tensors = [tfk.backend.expand_dims(t) for t in [l0, l1, l2, l3, l4]]
        # lab_batch_scaled = tfk.backend.concatenate(l_tensors, axis=2)

        target_size = np.sqrt(((lab_batch_scaled[:, :, 3]*0.2) ** 2) + ((lab_batch_scaled[:, :, 4]*0.2) ** 2)) # array
        target_x = np.reshape(lab_batch[:, :, 1], (1, np.prod(batch_size))) # xc
        target_y = np.reshape(lab_batch[:, :, 2], (1, np.prod(batch_size))) # yc
        targetoff_x = np.reshape(lab_batch[:, :, 3], (1, np.prod(batch_size))) # w
        targetoff_y = np.reshape(lab_batch[:, :, 4], (1, np.prod(batch_size))) # h

        target_x = tf.convert_to_tensor(target_x)
        target_y = tf.convert_to_tensor(target_y)
        targetoff_x = tf.convert_to_tensor(targetoff_x)
        targetoff_y = tf.convert_to_tensor(targetoff_y)

        # lab_batch_slices = tf.unstack(lab_batch, axis=2)
        #
        # target_x = tfk.backend.reshape(lab_batch_slices[1], shape=(1, np.prod(batch_size)))
        # target_y = tfk.backend.reshape(lab_batch_slices[2], shape=(1, np.prod(batch_size)))
        # targetoff_x = tfk.backend.reshape(lab_batch_slices[3], shape=(1, np.prod(batch_size)))
        # targetoff_y = tfk.backend.reshape(lab_batch_slices[4], shape=(1, np.prod(batch_size)))

        if(rand_loc):

            off_x = targetoff_x * (tfk.backend.random_uniform(tfk.backend.int_shape(targetoff_x), -0.4, 0.4))
            off_y = targetoff_y * (tfk.backend.random_uniform(tfk.backend.int_shape(targetoff_y), -0.4, 0.4))

            target_x = target_x + off_x
            target_y = target_y + off_y

        target_y = target_y - 0.05

        scale = target_size / float(current_patch_size)
        scale = tfk.backend.reshape(scale, shape=(1, anglesize))

        s = tfk.backend.int_shape(adv_batch) # 6 x 14 x 3 x 416 x 416
        # adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16
        # msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # 84 x 3 x 416 x 16

        adv_batch = tfk.backend.reshape(adv_batch, shape=(s[0] * s[1], s[2], s[3], s[4]))
        # print(tfk.backend.int_shape(adv_batch))
        msk_batch = tfk.backend.reshape(msk_batch, shape=(s[0] * s[1], s[2], s[3], s[4]))

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2

        sin = tfk.backend.sin(angle)
        cos = tfk.backend.cos(angle)

        # Theta = rotation, rescale matrix
        theta = tfk.backend.constant(0, shape=(anglesize,2,3)) # dim 84 x 2 x 3 (N x 2 x 3) required by F.affine_grid
        #print(tfk.backend.int_shape(theta))

        # tfk.backend.set_session(sess)
        # theta = sess.run(theta)
        # cos = sess.run(cos)
        # sin = sess.run(sin)
        # scale = sess.run(scale)
        # tx = sess.run(tx)
        # ty = sess.run(ty)

        theta = np.array(theta)
        cos = np.array(cos)
        sin = np.array(sin)
        scale = np.array(scale)
        tx = np.array(tx)
        ty = np.array(ty)

        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = ty * cos / scale + tx * sin / scale # todo I have swapped x and y wrt original code! don't know why still
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -ty * sin / scale + tx * cos / scale # todo I have swapped x and y wrt original code! don't know why still

        theta = tf.convert_to_tensor(theta)

        # theta_rows = tf.unstack(theta, axis=1) # dim 84 x 3 each NB list!
        # #print(tfk.backend.int_shape(theta_rows[0]))
        #
        # theta_row1_el = tf.unstack(theta_rows[0], axis = 1) # dim 84 each NB list!
        # #print(tfk.backend.int_shape(theta_row1_el[0]))
        # theta_row2_el = tf.unstack(theta_rows[1], axis=1) # dim 84 each NB list!
        #
        # theta_row1_el[0] = cos/scale
        # theta_row1_el[1] = sin/scale
        # theta_row1_el[2] = tx*cos/scale+ty*sin/scale
        #
        # theta_row2_el[0] = -sin/scale
        # theta_row2_el[1] = cos/scale
        # theta_row2_el[2] = -tx*sin/scale+ty*cos/scale
        #
        # theta_row1 = tfk.backend.concatenate(theta_row1_el, axis=0) # dim 84 x 3
        # print(tfk.backend.int_shape(theta_row1))
        # theta_row2 = tfk.backend.concatenate(theta_row2_el, axis=0) # dim 84 x 3
        # print(tfk.backend.int_shape(theta_row2))
        #
        # theta_row1 = tf.transpose(theta_row1, [1, 0])
        # print(tfk.backend.int_shape(theta_row1))
        # theta_row2 = tf.transpose(theta_row2, [1, 0])
        # print(tfk.backend.int_shape(theta_row2))
        #
        # #print(tfk.backend.int_shape(theta_row1))
        #
        # theta = tf.stack([theta_row1, theta_row2], axis=2)
        # print(tfk.backend.int_shape(theta))
        # theta = tf.transpose(theta, [0, 2, 1])
        # print(tfk.backend.int_shape(theta))
        # #print(tfk.backend.int_shape(theta))

        # # todo support - with pytorch predefined functions for affine
        # tfk.backend.set_session(sess)
        # theta = sess.run(theta)
        # adv_batch = sess.run(adv_batch)
        # msk_batch = sess.run(msk_batch)
        #
        # theta = torch.from_numpy(theta)
        # adv_batch = torch.from_numpy(adv_batch)
        # msk_batch = torch.from_numpy(msk_batch)
        #
        # grid = F.affine_grid(theta, adv_batch.shape)  # adv_batch should be of type N x C x Hin x Win. Output is N x Hg x Wg x 2
        #
        # adv_batch_t = F.grid_sample(adv_batch, grid)  # computes the output using input values and pixel locations from grid.
        # msk_batch_t = F.grid_sample(msk_batch, grid)  # Output has dim N x C x Hg x Wg

        # todo with tensorflow functions for affine
        adv_batch = tf.transpose(adv_batch, [0, 3, 2, 1]) # make channel last for stn method (b X 416 X 416 X 3)
        msk_batch = tf.transpose(msk_batch, [0, 3, 2, 1])

        adv_batch_t = stn.spatial_transformer_network(adv_batch, theta)
        msk_batch_t = stn.spatial_transformer_network(msk_batch, theta)

        adv_batch_t = tf.transpose(adv_batch_t, [0, 3, 2, 1])
        msk_batch_t = tf.transpose(msk_batch_t, [0, 3, 2, 1])

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

        adv_batch_t = tfk.backend.reshape(adv_batch_t, shape=(s[0], s[1], s[2], s[3], s[4]))
        msk_batch_t = tfk.backend.reshape(msk_batch_t, shape=(s[0], s[1], s[2], s[3], s[4]))

        adv_batch_t = tfk.backend.clip(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        #img.show()
        #exit()

        # print((adv_batch_t * msk_batch_t).size()) dim = 6 x 14 x 3 x 416 x 416

        return adv_batch_t * msk_batch_t  # It is as if I have passed adv_batch_t "filtered" by the mask itself

# NB output of PatchTransformer is the input of PatchApplier

class PatchApplier():
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        img_batch = img_batch / 255
        advs = tf.unstack(adv_batch, axis=1)  # Returns a list of all slices along a given dimension, already without it.

        for adv in advs:

            img_batch = tf.where((tf.equal(adv, 0)), img_batch, adv)  # the output tensor has elements belonging to img_batch if adv == 0, else belonging to adv

        return img_batch*255

class MaxProbExtractor():
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
        # get values necessary for transformation



        if tfk.backend.ndim(YOLOoutput) == 3:
            YOLOoutput = tfk.backend.expand_dims(YOLOoutput, 0)  # add one dimension of size 1
        batch = tfk.backend.int_shape(YOLOoutput)[0] # if ndim == 4
        assert (tfk.backend.int_shape(YOLOoutput)[1] == (5 + self.num_cls ) * 5)  # the last 5 is the anchor boxes number after k-means clustering
                                                                # the first 5 is the number of parameters of each box: x, y, w, h, objectness score
                                                                # self.num_cls indicates class probabilities, i.e. 20 values for VOC and 80 for COCO
                                                                # in total, there are 125 parameters per grid cell when VOC, 425 when COCO
        h = tfk.backend.int_shape(YOLOoutput)[2] # 13
        w = tfk.backend.int_shape(YOLOoutput)[3] # 13

        # transform the output tensor from [batch, 425, 13, 13] to [batch, 80, 845]
        output = tfk.backend.reshape(YOLOoutput, shape=(batch, 5, 5 + self.num_cls , h * w)) # [batch, 5, 85, 169]
        output = tf.transpose(output, [0, 2, 1, 3]) # [batch, 85, 5, 169] swap 5 and 85, in position 1 and 2 respectively
        output = tfk.backend.reshape(output, shape=(batch, 5 + self.num_cls , 5 * h * w)) # [batch, 85, 1805]

        # sess = tf.Session()
        # tfk.backend.set_session(sess)
        # output_vec = sess.run(output)
        # print(output_vec[:,4,:])
        # print(output_vec[:,5,:])

        output_objectness = tfk.backend.sigmoid(output[:, 4, :])  # [batch, 845]  # iou_truth_pred in yolov1 paper --> objectness score ?
        output = output[:, 5:5 + self.num_cls , :]  # [batch, 80, 845]  # 845 = 5 * h * w

        # output_objectness = tfk.backend.sigmoid(output[:, 4:5, :])  # [batch, 845]  # iou_truth_pred in yolov1 paper --> objectness score ?
        # output = output[:, 5:, :]  # [batch, 80, 845]  # 845 = 5 * h * w

        # perform softmax to normalize probabilities for object classes to [0,1] along the 1st dim of size 80 (no. classes in COCO)
        normal_confs = tfk.backend.softmax(output, axis=1)
        # NB Softmax is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1

        # prod = output_objectness_2 * normal_confs
        # prod = tfk.backend.max(prod, 1)
        # prod = tfk.backend.max(prod, 1)

        # we only care for probabilities of the class of interest (person, i.e. cls_id = 0 in COCO)
        confs_for_class = normal_confs[:, self.cls_id, :]

        #confs_if_object = output_objectness  # ?
        #confs_if_object = confs_for_class * output_objectness  #confs_for_class * output_objectness  # ?
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)  # loss_target in patch_config

        # sess = tf.Session()
        # tfk.backend.set_session(sess)
        # prod = sess.run(prod)
        # output_objectness = sess.run(output_objectness)
        # confs_for_class = sess.run(confs_for_class)
        # confs_if_object = sess.run(confs_if_object)
        # print('output_objectness: ' + str(output_objectness))
        # print('confs_class: ' + str(confs_for_class))
        # print('confs_prod: ' + str(confs_if_object))
        # print('prod: ' + str(prod))

        # find the max probability for person
        max_conf = tfk.backend.max(confs_if_object, axis=1)

        # max_conf = sess.run(max_conf)
        # print('max_conf: ' + str(max_conf))
        # #max_conf_idx = tfk.backend.argmax(confs_if_object, axis=1)
        # # max_conf_idx is the index of the max (argmax)
        #
        # max_conf = tf.convert_to_tensor(max_conf)

        # todo esperimento
        # output = tfk.backend.reshape(YOLOoutput, [batch, h, w, 5, self.num_cls + 5])
        # box_confidence = tfk.backend.sigmoid(output[..., 4])
        # box_class_probs = tfk.backend.softmax(output[..., 5:5+self.num_cls])
        # box_class_probs = box_class_probs[..., 5]
        #
        # prod = box_confidence*box_class_probs
        # prod = tfk.backend.max(prod, -1)
        # prod = tfk.backend.max(prod, -1)
        # prod = tfk.backend.max(prod, -1)
        #
        # sess = tf.Session()
        # tfk.backend.set_session(sess)
        # prod = sess.run(prod)
        # print(prod)

        return max_conf # prod

class NPSCalculator():
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        # self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

        #self.printability_array = tf.get_variable(name="self.get_printability_array(printability_file, patch_side)", trainable=False)
        self.printability_array = self.get_printability_array(printability_file, patch_side)

        # get variable from self.printability_array and not train it
        # NB in tensorflow, this passage is not needed, since self.printability array is a tensor and is not considered in training, being not a variable
        # sess = tf.Session()
        # tfk.backend.set_session(sess)
        # self.printability_array = tf.Variable(self.printability_array, trainable=True)
        # sess.run(tf.global_variables_initializer())

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference

        color_dist = (adv_patch - self.printability_array+0.000001)
        #print(tfk.backend.int_shape(color_dist))
        color_dist = color_dist ** 2  # squared difference
        color_dist = tfk.backend.sum(color_dist, 1)+0.000001
        #print(tfk.backend.int_shape(color_dist))
        color_dist = tfk.backend.sqrt(color_dist)

        # only work with the min distance
        color_dist_prod = tfk.backend.min(color_dist, axis=0) #test: change prod for min (find distance to closest color)
        #print(type(color_dist_prod))
        print(tfk.backend.int_shape(color_dist_prod))

        # calculate the nps by summing over all pixels
        nps_score = tfk.backend.sum(color_dist_prod,0)
        nps_score = tfk.backend.sum(nps_score,0)
        return nps_score/tf.size(input=adv_patch, out_type=tf.float32)  # divide by the total number of elements in the input tensor

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
        pa = tf.convert_to_tensor(printability_array) # Creates a Tensor from a numpy array.
        return pa


class TotalVariation():
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # compute total variation of the adv_patch
        tvcomp1 = tfk.backend.sum(tfk.backend.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)  # NB -1 indicates the last element!
        tvcomp1 = tfk.backend.sum(tfk.backend.sum(tvcomp1,0),0)

        tvcomp2 = tfk.backend.sum(tfk.backend.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = tfk.backend.sum(tfk.backend.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/tf.size(input=adv_patch, out_type=tf.float32)


#TODO ____________________________________________________________________________________________________________________________________________________________
    # TODO Summary of PatchTransformer + PatchApplier:
    # take a batch of 6 images, consider one. For it, I have 14 ready adv patches, of which a number that varies for each image is non-zero (remember:
    # the mask is done starting from 0 and 1 labels in lab_batch. Suppose that 5 are non zero. It means that they correspond to 5 detected object in that image.
    # They are already transformed according to correct positions and scales of the 5 detected rectangles. Now, we consider the image of the six composing the batch,
    # and substitute the patches in their positions where they are not zero (so 5 out of 14 in this example)
#TODO ____________________________________________________________________________________________________________________________________________________________

'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''

# old for pytorch version

class InriaDataset(tfk.utils.Sequence):
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

    def __init__(self, img_dir, lab_dir, max_lab, imgsize): # imgsize is 416 from yolo
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
        #self.shuffle = shuffle
        #self.batch_size = batch_size
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

        # open image
        image = Image.open(img_path).convert('RGB') # image with still its own dimensions
        w, h = image.size
        dim_orig = [w, h]

        if os.path.getsize(lab_path):       # check to see if label file contains data.
            label = np.loadtxt(lab_path)    # load data from a textfile
        else:
            label = np.ones([5])

        label = tf.convert_to_tensor(label)
        label = tf.cast(label, tf.float32)
        #if label.dim() == 1:
        if tfk.backend.ndim(label) == 1:
            label = tfk.backend.expand_dims(label, 0)

        image, label = self.pad_and_scale(image, label)
        image = tfk.preprocessing.image.img_to_array(image) # channel last
        image = np.rollaxis(image, 2, 0) # channel first

        label = self.pad_lab(label)  # to make it agree with max_lab dimensions. We choose a max_lab to say: no more than 14 persons could stand in one picture
        tuple_out = (image, label, dim_orig)

        return tuple_out

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     img_batch = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     lab_batch = np.empty((self.batch_size), dtype=int)
    #
    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')
    #
    #         # Store class
    #         y[i] = self.labels[ID]
    #
    #     return X, tfk.utils.to_categorical(y, num_classes=self.n_classes)

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        #label = tf.Variable(lab)
        w,h = img.size # sizes of my image
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                # lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                # lab[:, [3]] = (lab[:, [3]] * w / h)

                # Method 1) manual split + make list + concatenate
                l0 = lab[:, 0]
                l1 = (lab[:, 1] * w + padding) / h
                l2 = lab[:, 2]
                l3 = (lab[:, 3] * w / h)
                l4 = lab[:, 4]

                l_tensors = [tfk.backend.expand_dims(t) for t in [l0, l1,l2,l3,l4]]
                label = tfk.backend.concatenate(l_tensors, axis=1)

                # # Method 2) unstack + concatenate
                #
                # lab_slices = tf.unstack(lab, axis=1)
                # lab_slices[0] = lab[:, 0]
                # lab_slices[1] = (lab[:, 1] * w + padding) / h
                # lab_slices[2] = lab[:, 2]
                # lab_slices[3] = (lab[:, 3] * w / h)
                # lab_slices[4] = lab[:, 4]
                #
                # label = tfk.backend.concatenate(lab_slices, axis=1)

            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                # lab[:, [4]] = (lab[:, [4]] * h  / w)

                # Method 1) manual split + make list + concatenate
                l0 = lab[:, 0]
                l1 = lab[:, 1]
                l2 = (lab[:, 2] * h + padding) / w
                l3 = lab[:, 3]
                l4 = (lab[:, 4] * h  / w)

                l_tensors = [tfk.backend.expand_dims(t) for t in [l0, l1, l2, l3, l4]]
                label = tfk.backend.concatenate(l_tensors, axis=1)

                # # Method 2) unstack + concatenate
                #
                # lab_slices = tf.unstack(lab, axis=1)
                # lab_slices[0] = lab[:, 0]
                # lab_slices[0] = lab[:, 1]
                # lab_slices[0] = (lab[:, 2] * h + padding) / w
                # lab_slices[0] = lab[:, 3]
                # lab_slices[0] = (lab[:, 4] * h  / w)
                #
                # label = tfk.backend.concatenate(lab_slices, axis=1)

        # padded_img = tfk.backend.resize_images(padded_img, self.imgsize, self.imgsize, data_format="channels_first")
        img_dim = (self.imgsize, self.imgsize)
        padded_img = Image.Image.resize(padded_img, img_dim)

        return padded_img, label

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - tfk.backend.int_shape(lab)[0]
        if(pad_size>0):

            padding = tf.constant([[0, pad_size], [0, 0]]) # in vertical direction put nothing before tensor, pad_size after
                                                            # in horizontal direction put nothing before and after tensor
            padded_lab = tf.pad(lab, padding, 'CONSTANT', constant_values=1)
            # padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)  # padding of the labels to have a pad_size = max_lab (14 here).
                                                                     # add 1s to make dimensions = max_lab x batch_size (14 x 6) after the images lines,
                                                                     # whose number is not known a priori
        else:
            padded_lab = lab
        return padded_lab

