from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import os
import json
import torch
import numpy as np


net_type = 'mb2-ssd-lite'
model_path_selection = 'mb2-ssd-lite'

if model_path_selection == 'mb1-ssd':
    model_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/models/mbnt1_ssd_voc.pth"
elif model_path_selection == 'mb2-ssd-lite':
    model_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/models/mbnt2_ssd_lite_voc.pth"
elif model_path_selection == 'vgg16-ssd':
    model_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/models/vgg16-ssd-mp-0_7726.pth"

voc_label_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/voc_labels.txt" # VOC

class_names = [name.strip() for name in open(voc_label_path).readlines()]

# load net
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    #sys.exit(1)
net.load(model_path)

# call net predictor
if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

img_dir = "C:/Users/Alessandro/Desktop/SR _red/single_box_trial_ssdlite_mbntv2_coco/inria_few_data_img/"
destination_path = "C:/Users/Alessandro/Desktop/SR _red/single_box_trial_ssdlite_mbntv2_voc/ssd_lite_mnv2_voc_clean_fewdata-labels/"

n=1
clean_results = []
for image_file in os.listdir(img_dir):

    clean_image_name = image_file

    print('IMAGE #' + str(n) + ': ' + os.path.splitext(clean_image_name)[0])
    n+=1

    clean_label_name = image_file.replace('.png', '.txt')

    img_path = os.path.join(img_dir, clean_image_name)
    clean_label_path = os.path.join(destination_path, clean_label_name)

    orig_image = cv2.imread(img_path)

    h_orig = orig_image.shape[0] # for opencv it is (height, width) and .shape, while for PIL it is (width, height) and .size
    w_orig = orig_image.shape[1]

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    # NB boxes denormalized already in predictor.py lines 67-70

    boxes_array = boxes.numpy()

    if np.size(boxes_array, 0) != 0:
        labels_tmp = labels.unsqueeze(1)
        probs_tmp = probs.unsqueeze(1)
    else:
        labels_tmp = labels
        probs_tmp = probs

    labels_array = labels_tmp.numpy()
    probs_array = probs_tmp.numpy()

    if np.size(boxes_array, 0) != 0:
        results = np.concatenate((labels_array, boxes_array, probs_array), axis=1)
    else:
        results = np.concatenate((labels_array, boxes_array, probs_array), axis=0)

    if np.size(boxes_array, 0) != 0:
        results_list = np.split(results, np.size(boxes_array, 0), axis=0)
    else:
        results_list = []

    # plot boxes
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    2)  # line type

    # save detected image
    # destination = os.path.join(destination_path, 'after_detection/', clean_image_name)
    # cv2.imwrite(destination, orig_image)


    textfile = open(clean_label_path, 'w+')
    for res in results_list:
        res = np.squeeze(res, axis=0)
        cls_id = res[0]
        if (cls_id == 15):  # if person for VOC

            left, top, right, bottom = res[1:5] # being cv2.rectangle done with the top left and bottom right points coordinates

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h_orig, np.floor(bottom + 0.5).astype('int32'))
            right = min(w_orig, np.floor(right + 0.5).astype('int32'))

            width = right - left
            height = bottom - top
            x_center = left + width / 2
            y_center = top + height / 2

            # NB rescale xc, yc, box_w and box_h to original image dimensions
            textfile.write(
                f'{cls_id-15} {x_center / w_orig} {y_center / h_orig} {width / w_orig} {height / h_orig} {res[5]}\n')

            # clean_results.append({'image_id': os.path.splitext(clean_image_name)[0], 'bbox': [(x_center - width / 2) / w_orig,
            #                                                  (y_center - height / 2) / h_orig,
            #                                                  width / w_orig,
            #                                                  height / h_orig],
            #                       'score': res[5],
            #                       'category_id': 1})
            # print(clean_results)

    textfile.close()

# with open("C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/json_files/ssdvgg16/clean_results.json", 'w') as fp:
#     json.dump(clean_results, fp)

