#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


# todo NB ANNOTATIONS COINCIDES WITH CLEAN DETECTIONS! THEY ARE NOT THE GROUND-TRUTH OF INRIA DATASET, SO CLEAN RESULTS ARE 100% ASSUMED AS CORRECT
#annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssd_mobilenet_v1_coco_2017_11_17/clean/ssdmnv1-labels/', class_label_map={0: 'person'})
annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/inria_test_annotations_darknet/', class_label_map={0: 'person'})

clean_results = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenet_v1_coco_2017_11_17/clean_results.json', class_label_map={0: 'person'})
patch_results_cls_obj_1000_max = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patches/patch_results_objcls_1000_max.json", class_label_map={0: 'person'})
patch_results_cls_obj_1000_thresh = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patches/patch_results_objcls_1000_threshold.json", class_label_map={0: 'person'})
#patch_results_cls_obj_2000_max = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patches/patch_results_objcls_2000_max.json", class_label_map={0: 'person'})
patch_results_objonly_1000_max = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patches/patch_results_objonly_1000_max.json", class_label_map={0: 'person'})
patch_results_obj_1000_max_patchsize200 = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patches/patch_results_obj_1000_max_patchsize200.json", class_label_map={0: 'person'})

patch_random = bb.io.load('det_coco', "C:/Users\Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/patch_results_random.json", class_label_map={0: 'person'})


plt.figure()

clean = bb.stat.pr(clean_results, annotations, threshold=0.5)
print(clean)

obj_cls_1000_max = bb.stat.pr(patch_results_cls_obj_1000_max, annotations, threshold=0.5)
obj_cls_1000_thresh = bb.stat.pr(patch_results_cls_obj_1000_thresh, annotations, threshold=0.5)
#obj_cls_2000_max = bb.stat.pr(patch_results_cls_obj_2000_max, annotations, threshold=0.5)
obj_only_1000_max = bb.stat.pr(patch_results_objonly_1000_max, annotations, threshold=0.5)
obj_1000_max_patchsize200 = bb.stat.pr(patch_results_obj_1000_max_patchsize200, annotations, threshold=0.5)
random_noise = bb.stat.pr(patch_random, annotations, threshold=0.5)

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(clean)
plt.plot(clean['recall'], clean['precision'], label=f'CLEAN: AP: {round(ap*100, 2)}%') #, RECALL: {round(clean["recall"].iloc[-1]*100, 2)}%')

#ap = bb.stat.ap(obj_only)
#plt.plot(obj_only['recall'], obj_only['precision'], label=f'OBJ: AP: {round(ap*100, 2)}%, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

#ap = bb.stat.ap(cls_only)
#plt.plot(cls_only['recall'], cls_only['precision'], label=f'CLS: AP: {round(ap*100, 2)}%, RECALL: {round(cls_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_cls_1000_max)
plt.plot(obj_cls_1000_max['recall'], obj_cls_1000_max['precision'], label=f'OBJ-CLS_MAX_1K_P300: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_cls["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_cls_1000_thresh)
plt.plot(obj_cls_1000_thresh['recall'], obj_cls_1000_thresh['precision'], label=f'OBJ-CLS_THRESH_1K_P300: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_cls["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(obj_cls_2000_max)
# plt.plot(obj_cls_2000_max['recall'], obj_cls_2000_max['precision'], label=f'OBJ-CLS_MAX_2K_P300: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_cls["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_only_1000_max)
plt.plot(obj_only_1000_max['recall'], obj_only_1000_max['precision'], label=f'OBJ_MAX_1K_P300: AP: {round(ap*100, 2)}%')

ap = bb.stat.ap(obj_1000_max_patchsize200)
plt.plot(obj_1000_max_patchsize200['recall'], obj_1000_max_patchsize200['precision'], label=f'OBJ-CLS_MAX_1K_P200: AP: {round(ap*100, 2)}%')

#ap = bb.stat.ap(obj_only_paper)
#plt.plot(obj_only_paper['recall'], obj_only_paper['precision'], label=f'OBJ_PAPER: AP: {round(ap*100, 2)}%, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(random_noise)
plt.plot(random_noise['recall'], random_noise['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%') #, RECALL: {round(random_noise["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(random_image)
# plt.plot(random_image['recall'], random_image['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%, RECALL: {round(random_image["recall"].iloc[-1]*100, 2)}%')

plt.gcf().suptitle('SSDLite_MobileNetV2_COCO, PATCH:SSDLite_MobileNetV2_VOC')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/single_patches_vs_single_networks/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patch/pr_curve_ssdlitembnv2_ref_real_inria.png')
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/single_patches_vs_single_networks/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mbntv2_patch/pr_curve_ssdlitembnv2_ref_real_inria.eps')
#plt.show()

