from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import PIL
import os
import json
import torch
import numpy as np
from load_data import *
from torchvision import transforms
import matplotlib.pyplot as plt

def generate_patch(type, patch_size):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        adv_patch_cpu = torch.full((3, patch_size, patch_size), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((3, patch_size, patch_size))

    return adv_patch_cpu

def pad_and_scale(img, lab, common_size):  # this method for taking a non-square image and make it square by filling the difference in w and h with gray

    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h / w)
    resize = transforms.Resize((common_size, common_size))  # make a square image of dim 416 x 416
    padded_img = resize(padded_img)  # choose here
    return padded_img, lab

def remove_pad(w_orig, h_orig, in_img):

        w = w_orig
        h = h_orig

        img = transforms.ToPILImage('RGB')(in_img)

        dim_to_pad = 1 if w < h else 2

        if dim_to_pad == 1:
            padding = (h - w) / 2
            #padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            #padded_img.paste(img, (int(padding), 0))
            image = Image.Image.resize(img, (h, h))
            image = Image.Image.crop(image, (int(padding), 0, int(padding) + w, h))

        else:
            padding = (w - h) / 2
            # padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            # padded_img.paste(img, (0, int(padding)))
            image = Image.Image.resize(img, (w, w))
            image = Image.Image.crop(image, (0, int(padding), w, int(padding) + h))

        return image

net_type = 'mb1-ssd'
model_path_selection = 'mb1-ssd'

if model_path_selection == 'mb1-ssd':
    model_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/models/mbnt1_ssd_voc.pth"
elif model_path_selection == 'mb2-ssd-lite':
    model_path = "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/models/mbnt2_ssd_lite_voc.pth"

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
destination_path = "C:/Users/Alessandro/Desktop/SR _red/single_box_trial_ssd_mbntv1_voc_random/"
patchfile = "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2p_adv_COCO_thys2019/saved_patches_mytrial/ssdv1_ok/mbntv1ssd_max_objcls_meanstd.png"

square_size = 500
patch_size = 300

#patch_img = Image.open(patchfile).convert('RGB')
# plt.imshow(patch_img)
# plt.show()
#patch_img = patch_img.resize((patch_size,patch_size))
#adv_patch = transforms.ToTensor()(patch_img) # already in range 0,1
# print(adv_patch.type())
# print('adv_patch: ' + str(adv_patch))

adv_patch = generate_patch('gray', patch_size)

n=1
clean_results = []
for image_file in os.listdir(img_dir):

    clean_image_name = image_file

    print('\nIMAGE #' + str(n) + ': ' + os.path.splitext(clean_image_name)[0])
    n+=1

    name = os.path.splitext(clean_image_name)[0] # name without extension
    clean_label_name = image_file.replace('.png', '.txt')
    patch_label_name = name + '_p.txt'

    img_path = os.path.join(img_dir, clean_image_name)
    clean_label_path = os.path.join(destination_path, 'ssd_mnv1_voc_clean_fewdata-labels/', clean_label_name)

    # create folders if they do not exists
    single_img_labdir = os.path.join(destination_path, 'ssd_mnv1_voc_patched_fewdata_labels/', name, 'patch/')
    if os.path.isdir(single_img_labdir) == False:
        os.makedirs(single_img_labdir)


    #orig_image = cv2.imread(img_path) # numpy array
    orig_image = Image.open(img_path).convert('RGB')

    h_orig = orig_image.size[1] # for opencv it is (height, width) and .shape, while for PIL it is (width, height) and .size
    w_orig = orig_image.size[0]

    print('Start reading generated label file used for patch application')
    # read this label file back as a tensor
    textfile = open(clean_label_path, 'r')
    if os.path.getsize(clean_label_path):  # check to see if label file contains data.
        label = np.loadtxt(textfile)
        label_nopad = label
        # print(label.shape)
    else:
        label = np.ones([6])
        label_nopad = label

    if np.ndim(label) == 1:
        # label = label.unsqueeze(0)
        label = np.expand_dims(label, 0)
        label_nopad = label

    label = torch.from_numpy(label).float()
    label_nopad = torch.from_numpy(label_nopad).float()

    print('label file used for patch application read correctly')

    # start image preprocessing to apply patch

    print('Start image preprocessing')
    # convert image numpy array to torch tensor
    #image_clean_ref = torch.from_numpy(orig_image)
    # convert torch tensor to PIL image
    #image_clean_ref = transforms.ToPILImage('RGB')(image_clean_ref)
    image_clean_ref = orig_image

    image_p, label_p = pad_and_scale(image_clean_ref, label, common_size=square_size)

    # convert image back to torch tensor
    image_tens = transforms.ToTensor()(image_p)

    # add fake batch size, fake because it has size = 1, so it's a single image (i.e you don't really need)
    img_fake_batch = torch.unsqueeze(image_tens, 0)
    lab_fake_batch = torch.unsqueeze(label, 0)
    lab_fake_batch_nopad = label_nopad.unsqueeze(0)

    print('Clean label original: ' + str(lab_fake_batch.size()))

    # start splitting label clean file for single patch application
    lab_fake_batch_boxsplit = torch.unbind(lab_fake_batch, axis=1)  # to get each box line separated for patch application
    lab_fake_batch_boxsplit_nopad = lab_fake_batch_nopad.unbind(1)

    single_img_labdir_c = os.path.join(destination_path, 'ssd_mnv1_voc_patched_fewdata_labels/', name, 'clean/')
    if os.path.isdir(single_img_labdir_c) == False:
        os.makedirs(single_img_labdir_c)

    dict_imageperbox_list = []
    #im_outperbox_list = []
    k = 1

    for single_box_label, single_box_label_nopad in zip(lab_fake_batch_boxsplit, lab_fake_batch_boxsplit_nopad):

        single_box_clean_name = name + '_box' + str(k) + '-c.txt' # nome con cui salverò l'immagine pulita
        single_box_clean_path = os.path.join(single_img_labdir_c, single_box_clean_name)

        textfile_clean = open(single_box_clean_path, 'w+')
        textfile_clean.write(
            f'{single_box_label_nopad[0][0]} {single_box_label_nopad[0][1]} {single_box_label_nopad[0][2]} {single_box_label_nopad[0][3]} {single_box_label_nopad[0][4]} {single_box_label_nopad[0][5]}')

        single_box_label = torch.unsqueeze(single_box_label, axis=1)

        box_name = name + '_box' + str(k) + '.jpg'
        k += 1

        adv_batch_t = PatchTransformer()(adv_patch, single_box_label, square_size, do_rotate=True, rand_loc=False)
        p_img_batch = PatchApplier()(img_fake_batch, adv_batch_t)
        p_img = torch.squeeze(p_img_batch, 0)

        # come back to original dimensions
        p_img_orig = remove_pad(w_orig, h_orig, p_img)

        out_path_part = "C:/Users/Alessandro/Desktop/SR _red/single_box_trial_ssd_mbntv1_voc_random/patch_applied_images_no_det/"
        out_path = os.path.join(out_path_part, box_name)
        p_img_orig.save(out_path) # save patched image in folder
        # plt.imshow(p_img_orig)
        # plt.show()

        print('End image preprocessing')

        # load patched image from folder with cv2 to run inference
        image = cv2.imread(out_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # do inference
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        # print(boxes)
        # print(labels)
        # print(probs)
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

        #plot boxes
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # font scale
                        (255, 0, 255),
                        2)  # line type

        #save detected image
        destination = os.path.join(destination_path, 'patch_applied_images_det/', box_name)
        cv2.imwrite(destination, image)

        dict_imageperbox_list.append(results_list)

    for i, dict in enumerate(dict_imageperbox_list):
        box_name = name + '_box' + str(i + 1) + '.txt'

        txtpath_p = os.path.join(single_img_labdir, box_name)

        textfile = open(txtpath_p, 'w+')
        for res in dict:
            res = np.squeeze(res, axis=0)
            cls_id = res[0]
            if (cls_id == 15):  # if person

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

                clean_results.append({'image_id': os.path.splitext(clean_image_name)[0], 'bbox': [(x_center - width / 2) / w_orig,
                                                                 (y_center - height / 2) / h_orig,
                                                                 width / w_orig,
                                                                 height / h_orig],
                                      'score': res[5],
                                      'category_id': 1})
                # print(clean_results)

        textfile.close()

