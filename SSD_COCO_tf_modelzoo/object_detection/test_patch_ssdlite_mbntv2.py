#!/usr/bin/env python
# coding: utf-8

# # Object Detection API Demo
#
# <table align="left"><td>
#   <a target="_blank"  href="https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
#   </a>
# </td><td>
#   <a target="_blank"  href="https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb">
#     <img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
# </td></table>

# Welcome to the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image.

# > **Important**: This tutorial is to help you through the first step towards using [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to build models. If you just just need an off the shelf model that does the job, see the [TFHub object detection example](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb).

# # Setup

# Important: If you're running on a local machine, be sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). This notebook includes only what's necessary to run in Colab.

# ### Install

# In[ ]:


#get_ipython().system('pip install -U --pre tensorflow=="2.*"')

# Make sure you have `pycocotools` installed

# In[ ]:


#get_ipython().system('pip install pycocotools')

# Get `tensorflow/models` or `cd` to parent directory of the repository.

# In[ ]:


import os
import pathlib
import json
from load_data import *

# if "models" in pathlib.Path.cwd().parts:
#     while "models" in pathlib.Path.cwd().parts:
#         os.chdir('..')
# elif not pathlib.Path('models').exists():
#     get_ipython().system('git clone --depth 1 https://github.com/tensorflow/models')

# Compile protobufs and install the object_detection package

# In[ ]:


#get_ipython().run_cell_magic('bash', '', 'cd models/research/\nprotoc object_detection/protos/*.proto --python_out=.')

# In[ ]:


#get_ipython().run_cell_magic('bash', '', 'cd models/research\npip install .')

# ### Imports

# In[ ]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

# Import the object detection module.

# In[ ]:


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Patches:

# In[ ]:


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# ## Loader

# In[ ]:


def load_model():
    # base_url = 'http://download.tensorflow.org/models/object_detection/'
    # model_file = model_name + '.tar.gz'
    # model_dir = tf.keras.utils.get_file(
    #     fname=model_name,
    #     origin=base_url + model_file,
    #     untar=True)

    model_dir = 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/models_proto_converted/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/'

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# For the sake of simplicity we will test on 2 images:

# In[ ]:


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/inria_test_pos')
#PATH_TO_TEST_IMAGES_DIR_AFTER = pathlib.Path('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssd_mobilenetv1/clean')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))


# # Detection

# Load an object detection model:

# In[ ]:


detection_model = load_model()

# Check the model's input signature, it expects a batch of 3-color images of type uint8:

# In[ ]:


print(detection_model.inputs)

# And retuns several outputs:

# In[ ]:


detection_model.output_dtypes

# In[ ]:


print(detection_model.output_shapes)


# Add a wrapper function to call the model, and cleanup the outputs:

# In[ ]:

def generate_patch(type):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        # adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        adv_patch = tfk.backend.constant(0.5, dtype=tf.float32,
                                         shape=(3, patch_size, patch_size))

    elif type == 'random':
        # adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
        adv_patch = tfk.backend.random_uniform((3, patch_size, patch_size), 0, 1)

    return adv_patch


def pad_and_scale(model_image_size, img, lab):
    """

    Args:
        img:

    Returns:

    """
    # label = tf.Variable(lab)
    w, h = img.size  # sizes of my image

    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
            # lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            # lab[:, [3]] = (lab[:, [3]] * w / h)

            # Method 1) manual split + make list + concatenate
            l0 = lab[:, 0]
            l1 = (lab[:, 1] * w + padding) / h
            l2 = lab[:, 2]
            l3 = (lab[:, 3] * w / h)
            l4 = lab[:, 4]

            l_tensors = [tfk.backend.expand_dims(t) for t in [l0, l1, l2, l3, l4]]
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
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            # lab[:, [4]] = (lab[:, [4]] * h  / w)

            # Method 1) manual split + make list + concatenate
            l0 = lab[:, 0]
            l1 = lab[:, 1]
            l2 = (lab[:, 2] * h + padding) / w
            l3 = lab[:, 3]
            l4 = (lab[:, 4] * h / w)

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
    img_dim = (model_image_size, model_image_size)  # resize image to yolo dim (416 x 416)
    padded_img = Image.Image.resize(padded_img, img_dim)

    #padded_img.save(os.path.join(savedir, "proper_patched_obj/", "debug/", properpatchedname))

    return padded_img, label


def remove_pad(w_orig, h_orig, img):
    w = w_orig
    h = h_orig

    dim_to_pad = 1 if w < h else 2

    if dim_to_pad == 1:
        padding = (h - w) / 2
        # padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
        # padded_img.paste(img, (int(padding), 0))
        image = Image.Image.resize(img, (h, h))
        image = Image.Image.crop(image, (int(padding), 0, int(padding) + w, h))

    else:
        padding = (w - h) / 2
        # padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
        # padded_img.paste(img, (0, int(padding)))
        image = Image.Image.resize(img, (w, w))
        image = Image.Image.crop(image, (0, int(padding), w, int(padding) + h))

    return image


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


# Run it on each test image and show the results:

# In[ ]:


def patch_and_show_inference(model, image_path, label):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = Image.open(image_path)
    image_dims = image.size

    image_pad, label = pad_and_scale(model_image_size, image, label)

    img_arr = tfk.preprocessing.image.img_to_array(image_pad, data_format='channels_first')
    image_tens = tf.convert_to_tensor(img_arr) # from array to tensor
    #image_tens = tf.transpose(image_tens, [2, 1, 0])
    #print(tfk.backend.int_shape(image_tens)) # 3 x 416 x 416

    # add fake batch size, fake because it has size = 1, so it's a single image (i.e you don't really need)
    img_fake_batch = tfk.backend.expand_dims(image_tens, 0)  # already channels_first
    lab_fake_batch = tfk.backend.expand_dims(label, 0)

    adv_batch_t = patch_transformer.forward(adv_patch, lab_fake_batch, model_image_size, do_rotate=True, rand_loc=False)
    p_img_batch = patch_applier.forward(img_fake_batch, adv_batch_t)
    p_img = tfk.backend.squeeze(p_img_batch, 0)

    p_img = np.array(p_img) # from tensor to array
    p_img = tfk.preprocessing.image.array_to_img(p_img, data_format='channels_first') # from array to image
    p_img_orig = remove_pad(image_dims[0], image_dims[1], p_img)  # this is an image

    #p_img_orig.save(os.path.join(savedir, "proper_patched_paper_obj/", "bef_detection/", properpatchedname))

    image_np = np.array(p_img_orig) # convert image to array

    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    im_out, dict = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    return im_out, dict, image_dims
    #display(Image.fromarray(image_np))


# In[ ]:
savedir = "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssdlite_mobilenet_v2_coco_2018_05_09/"
patchfile = "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2p_adv_COCO_thys2019/saved_patches_mytrial/ssdlitev2_ok/mbntv2_ssdlite_max_objonly_1000epochs_patchsize200_meanstd.jpg"
txtpath_c = "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssdlite_mobilenet_v2_coco_2018_05_09/clean/ssd_lite_mnv2-labels/"

patch_applier = PatchApplier()
patch_transformer = PatchTransformer()
batch_size = 1
max_lab = 14
model_image_size = 416

patch_size = 300

# todo OPEN PATCH IMAGE FILE
# patch_img = Image.open(patchfile).convert('RGB')
# patch_img = patch_img.resize((patch_size,patch_size))
# # plt.imshow(patch_img)
# # plt.show()
#
# adv_patch = tfk.preprocessing.image.img_to_array(patch_img, data_format='channels_first') # to np array
# adv_patch = adv_patch / 255
# adv_patch = tf.convert_to_tensor(adv_patch) # to tf tensor

adv_patch = generate_patch("gray")

patch_results = []
n = 0
for image_path in TEST_IMAGE_PATHS:
    image_path_str = str(image_path)

    n += 1
    print('\nIMAGE NO.' + str(n))

    if image_path_str.endswith('.jpg') or image_path_str.endswith('.png'):
        name = os.path.splitext(image_path_str)[0]  # image name w/o extension
        name = os.path.split(name)[1]

        properpatchedname = name + "_p.png"
        txtname = properpatchedname.replace('.png', '.txt')

        txtname_clean = name + ".txt"
        txtpath_clean = os.path.join(txtpath_c, txtname_clean)
        txtpath_p = os.path.join(savedir, 'proper_patched_random/', 'ssd_lite_mnv2-labels/', txtname)

    print('Start reading generated label file used for patch application')
    textfile = open(txtpath_clean, 'r')
    if os.path.getsize(txtpath_clean):  # check to see if label file contains data.
        label = np.loadtxt(textfile)
        # print(label.shape)
    else:
        label = np.ones([5])

    label = tf.convert_to_tensor(label, dtype=tf.float32)

    if tfk.backend.ndim(label) == 1:
        # label = label.unsqueeze(0)
        label = tfk.backend.expand_dims(label, 0)
    print('label file used for patch application read correctly')

    im_out, dict, image_dims = patch_and_show_inference(detection_model, image_path, label)

    im_out = Image.fromarray(im_out)
    #im_out.save(os.path.join(savedir, "proper_patched_ale_obj_cls/", "after_detection/", properpatchedname))

    results_list = []
    for box in dict:
        out_vector = np.zeros((1, 6))

        out_vector[0][0] = dict[box][0][0]
        out_vector[0][1] = box[0]
        out_vector[0][2] = box[1]
        out_vector[0][3] = box[2]
        out_vector[0][4] = box[3]
        out_vector[0][5] = dict[box][0][1]

        results_list.append(out_vector)

    #print(results_list)

    textfile = open(txtpath_p, 'w+')
    for res in reversed(results_list):
        res = np.squeeze(res, axis=0)
        cls_id = res[0]
        if (cls_id == 0):  # if person

            # w_orig = image_dims[0]
            # h_orig = image_dims[1]

            top, left, bottom, right = res[1:5]

            # top = max(0, np.floor(top + 0.5).astype('int32'))
            # left = max(0, np.floor(left + 0.5).astype('int32'))
            # bottom = min(image_dims[1], np.floor(bottom + 0.5).astype('int32'))
            # right = min(image_dims[0], np.floor(right + 0.5).astype('int32'))

            width = right - left
            height = bottom - top
            x_center = left + width / 2
            y_center = top + height / 2

            # print((left, top), (right, bottom))
            # print((x_center, y_center), (width, height))
            #print((x_center, y_center), (width, height))

            # NB rescale xc, yc, box_w and box_h to original image dimensions
            textfile.write(
                f'{cls_id} {x_center} {y_center} {width} {height} \n')

            patch_results.append({'image_id': name, 'bbox': [(x_center - width / 2),
                                                             (y_center - height / 2),
                                                             width,
                                                             height],
                                  'score': res[5],
                                  'category_id': 1})
            # print(clean_results)

    textfile.close()

with open('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/patch_results_random.json', 'w') as fp:
    json.dump(patch_results, fp)


