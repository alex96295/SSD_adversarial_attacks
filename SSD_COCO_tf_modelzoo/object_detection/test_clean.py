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
    #     #     fname=model_name,
    #     #     origin=base_url + model_file,
    #     #     untar=True)

    model_dir = 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/models_proto_converted/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model'

    model = tf.saved_model.load(model_dir)
    model = model.signatures['serving_default']

    return model


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


# List of the strings that is used to add correct label for each box --> classes names
PATH_TO_LABELS = 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# For the sake of simplicity we will test on 2 images:

# In[ ]:


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path("C:/Users/Alessandro/Desktop/single_box_trial_ssdlite_mbntv2_coco/inria_few_data_img/")
#PATH_TO_TEST_IMAGES_DIR_AFTER = pathlib.Path('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssd_mobilenetv1/clean')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))


# # Detection

# Load an object detection model:

# In[ ]:


#model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model()

# Check the model's input signature, it expects a batch of 3-color images of type uint8:

# In[ ]:


#print(detection_model.inputs)

# And retuns several outputs:

# In[ ]:


#print(detection_model.output_dtypes)

# In[ ]:


#detection_model.output_shapes


# Add a wrapper function to call the model, and cleanup the outputs:

# In[ ]:


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)
    print(output_dict)

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


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image = Image.open(image_path)
    image_dims = image.size
    #print(image_dims)



    image_np = np.array(image)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    print(type(output_dict))
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
savedir = "C:/Users/Alessandro/Desktop/single_box_trial_ssdlite_mbntv2_coco/ssd_lite_mnv2_clean_fewdata-labels/"

clean_results = []
n = 0

for image_path in TEST_IMAGE_PATHS:
    image_path_str = str(image_path)

    n += 1
    print('\nIMAGE NO.' + str(n))

    if image_path_str.endswith('.jpg') or image_path_str.endswith('.png'):

        name = os.path.splitext(image_path_str)[0]
        name = os.path.split(name)[1]  # image name w/o extension
        #print(name)
        cleanname = name + ".png"

        txtname = name + '.txt'
        txtpath_c = os.path.join(savedir, txtname)



    im_out, dict, image_dims = show_inference(detection_model, image_path)

    im_out = Image.fromarray(im_out)
    #im_out.save(os.path.join(savedir, "clean/after_detection/", cleanname))

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

    textfile = open(txtpath_c, 'w+')
    for res in reversed(results_list):
        res = np.squeeze(res, axis=0)
        cls_id = res[0]
        if (cls_id == 0):  # if person

            # w_orig = image_dims[0]
            # h_orig = image_dims[1]

            top, left, bottom, right = res[1:5] # should be correct being the drawing of rectangles ymin, xmin, ymax, xmax (visualization_utils line 856)

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
                f'{cls_id} {x_center} {y_center} {width} {height} {res[5]}\n')

            clean_results.append({'image_id': name, 'bbox': [(x_center - width / 2),
                                                             (y_center - height / 2),
                                                             width,
                                                             height],
                                  'score': res[5],
                                  'category_id': 1})
            # print(clean_results)

    textfile.close()

# with open('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv2_coco_quantized/clean_results.json', 'w') as fp:
#     json.dump(clean_results, fp)


