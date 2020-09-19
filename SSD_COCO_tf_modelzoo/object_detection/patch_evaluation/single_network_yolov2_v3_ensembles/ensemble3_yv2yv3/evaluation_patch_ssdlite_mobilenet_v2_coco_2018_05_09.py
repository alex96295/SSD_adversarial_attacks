#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


# todo NB ANNOTATIONS COINCIDES WITH CLEAN DETECTIONS! THEY ARE NOT THE GROUND-TRUTH OF INRIA DATASET, SO CLEAN RESULTS ARE 100% ASSUMED AS CORRECT
#annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssdlite_mobilenet_v2_coco_2018_05_09/clean/ssd_lite_mnv2-labels/', class_label_map={0: 'person'})
annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/inria_test_annotations_darknet/', class_label_map={0: 'person'})

clean_results = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/clean_results.json', class_label_map={0: 'person'})
# patch_results_obj = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ensemble_yv2_yv3_ssdlitembntv2_patches/patch_results_obj.json', class_label_map={0: 'person'})
# patch_results_cls = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ensemble_yv2_yv3_ssdlitembntv2_patches/patch_results_cls.json', class_label_map={0: 'person'})
patch_results_cls_obj = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/ensemble_yv2_yv3_ssdlitembntv2_patches/patch_results_obj_cls.json", class_label_map={0: 'person'})
# patch_results_obj_paper = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/yolov3_patches/patch_results_obj_paper.json', class_label_map={0: 'person'})

# patch_results_random_noise = bb.io.load('det_coco', './json_files/yolov3/random_results.json', class_label_map={0: 'person'})
# patch_results_random_image = bb.io.load('det_coco', './json_files/yolov3/random_image_results.json', class_label_map={0: 'person'})

# In[8]:


plt.figure()

clean = bb.stat.pr(clean_results, annotations, threshold=0.5)
# obj_only = bb.stat.pr(patch_results_obj, annotations, threshold=0.5)
# cls_only = bb.stat.pr(patch_results_cls, annotations, threshold=0.5)
obj_cls = bb.stat.pr(patch_results_cls_obj, annotations, threshold=0.5)
# obj_only_paper = bb.stat.pr(patch_results_obj_paper, annotations, threshold=0.5)

#random_noise = bb.stat.pr(patch_results_random_noise, annotations, threshold=0.5)
#random_image = bb.stat.pr(patch_results_random_image, annotations, threshold=0.5)

#ap = bbb.ap(teddy[0], teddy[1])
#plt.plot(teddy[1], teddy[0], label=f'Teddy: mAP: {round(ap*100, 2)}%')

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(clean)
plt.plot(clean['recall'], clean['precision'], label=f'CLEAN: AP: {round(ap*100, 2)}%') #, RECALL: {round(clean["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(obj_only)
# plt.plot(obj_only['recall'], obj_only['precision'], label=f'OBJ: AP: {round(ap*100, 2)}%, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(cls_only)
# plt.plot(cls_only['recall'], cls_only['precision'], label=f'CLS: AP: {round(ap*100, 2)}%, RECALL: {round(cls_only["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_cls)
plt.plot(obj_cls['recall'], obj_cls['precision'], label=f'OBJ-CLS: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_cls["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(obj_only_paper)
# plt.plot(obj_only_paper['recall'], obj_only_paper['precision'], label=f'OBJ_PAPER: AP: {round(ap*100, 2)}%, RECALL: {round(obj_only["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(random_noise)
# plt.plot(random_noise['recall'], random_noise['precision'], label=f'NOISE: AP: {round(ap*100, 2)}%, RECALL: {round(random_noise["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(random_image)
# plt.plot(random_image['recall'], random_image['precision'], label=f'RAND_IMG: AP: {round(ap*100, 2)}%, RECALL: {round(random_image["recall"].iloc[-1]*100, 2)}%')

plt.gcf().suptitle('SSDLite_MobileNetV2, dataset = INRIA, PATCH:yv2+yv3+mbntv2')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/single_patches_vs_single_networks/ssdlite_mobilenet_v2_coco_2018_05_09/ensemble3_yv2_yv3_mbnt2/pr_curve_ssdlitembnv2_ref_real_inria.eps')
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/single_patches_vs_single_networks/ssdlite_mobilenet_v2_coco_2018_05_09/ensemble3_yv2_yv3_mbnt2/pr_curve_ssdlitembnv2_ref_real_inria.png')
#plt.show()

