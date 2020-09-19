#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


#annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/ssd_mobilenet_v1_coco_2017_11_17/clean/ssdmnv1-labels/', class_label_map={0: 'person'})
annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/inria_test_annotations_darknet/', class_label_map={0: 'person'})


# patch_obj_ale results on inria
patch_results_obj_ale_yolov2 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/yolov2_pytorch/json_files/yolov2_patches/patch_results_obj.json', class_label_map={0: 'person'})
patch_results_obj_ale_yolov3 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/yolov3_pytorch/json_files/yolov2_patches/proper_patched_obj.json', class_label_map={0: 'person'})
patch_results_obj_ale_yolov4 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/YOLOv4_d2p_adv_COCO/json_files/yolov2_patches/patch_results_obj.json', class_label_map={0: 'person'})

patch_results_obj_ale_ssdmbntv1 = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenet_v1_coco_2017_11_17/yolov2_patches/patch_results_obj_ale.json', class_label_map={0: 'person'})
patch_results_obj_ale_ssdmbntv2 = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenet_v2_coco_2018_03_29/yolov2_patches/patch_results_obj_ale.json', class_label_map={0: 'person'})
patch_results_obj_ale_ssdmbntv1_q = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv1_coco_quantized/yolov2_patches/patch_results_obj.json", class_label_map={0: 'person'})
patch_results_obj_ale_ssdmbntv2_q = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv2_coco_quantized/yolov2_patches/patch_results_obj.json', class_label_map={0: 'person'})

patch_results_obj_ale_ssdlitembntv2 = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/yolov2_patches/patch_results_obj_ale.json', class_label_map={0: 'person'})
patch_results_obj_ale_ssdmbntv3small = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv3_small/yolov2_patches/patch_results_obj_ale.json', class_label_map={0: 'person'})
patch_results_obj_ale_ssdmbntv3large = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv3_large/yolov2_patches/patch_results_obj_ale.json', class_label_map={0: 'person'})

patch_results_obj_ale_ssdmbntv1_voc = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/json_files/ssdmbntv1/yolov2_patches/patch_results_obj.json", class_label_map={0: 'person'})
patch_results_obj_ale_ssdlitembntv2_voc = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/json_files/ssdlitembntv2/yolov2_patches/patch_results_obj.json", class_label_map={0: 'person'})

plt.figure()

obj_ale_yolov2 = bb.stat.pr(patch_results_obj_ale_yolov2, annotations, threshold=0.5)
obj_ale_yolov3 = bb.stat.pr(patch_results_obj_ale_yolov3, annotations, threshold=0.5)
obj_ale_yolov4 = bb.stat.pr(patch_results_obj_ale_yolov4, annotations, threshold=0.5)
#obj_ale_yolov3_tiny = bb.stat.pr(patch_results_obj_ale_yolov3_tiny, annotations, threshold=0.5)

obj_ale_ssdmbntv1 = bb.stat.pr(patch_results_obj_ale_ssdmbntv1, annotations, threshold=0.5)
obj_ale_ssdmbntv2 = bb.stat.pr(patch_results_obj_ale_ssdmbntv2, annotations, threshold=0.5)
obj_ale_ssdlitembntv2 = bb.stat.pr(patch_results_obj_ale_ssdlitembntv2, annotations, threshold=0.5)
obj_ale_ssdmbntv1_q = bb.stat.pr(patch_results_obj_ale_ssdmbntv1_q, annotations, threshold=0.5)
obj_ale_ssdmbntv2_q = bb.stat.pr(patch_results_obj_ale_ssdmbntv2_q, annotations, threshold=0.5)

obj_ale_ssdmbntv3small = bb.stat.pr(patch_results_obj_ale_ssdmbntv3small, annotations, threshold=0.5)
obj_ale_ssdmbntv3large = bb.stat.pr(patch_results_obj_ale_ssdmbntv3large, annotations, threshold=0.5)

obj_ale_ssdlitembntv2_voc = bb.stat.pr(patch_results_obj_ale_ssdlitembntv2_voc, annotations, threshold=0.5)
obj_ale_ssdmbntv1_voc = bb.stat.pr(patch_results_obj_ale_ssdmbntv1_voc, annotations, threshold=0.5)

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ap = bb.stat.ap(obj_ale_yolov2)
plt.plot(obj_ale_yolov2['recall'], obj_ale_yolov2['precision'], label=f'YOLOV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_yolov3)
plt.plot(obj_ale_yolov3['recall'], obj_ale_yolov3['precision'], label=f'YOLOV3: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov3["recall"].iloc[-1]*100, 2)}%')

#ap = bb.stat.ap(obj_ale_yolov3_tiny)
#plt.plot(obj_ale_yolov3_tiny['recall'], obj_ale_yolov3_tiny['precision'], label=f'YOLOV3_TINY: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov3_tiny["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_yolov4)
plt.plot(obj_ale_yolov4['recall'], obj_ale_yolov4['precision'], label=f'YOLOV3: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov3["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv1)
plt.plot(obj_ale_ssdmbntv1['recall'], obj_ale_ssdmbntv1['precision'], label=f'SSD_MNV1: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv1["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv2)
plt.plot(obj_ale_ssdmbntv2['recall'], obj_ale_ssdmbntv2['precision'], label=f'SSD_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdlitembntv2)
plt.plot(obj_ale_ssdlitembntv2['recall'], obj_ale_ssdlitembntv2['precision'], label=f'SSDLITE_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv1_q)
plt.plot(obj_ale_ssdmbntv1_q['recall'], obj_ale_ssdmbntv1_q['precision'], label=f'SSD_MNV1_Q: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv2_q)
plt.plot(obj_ale_ssdmbntv2_q['recall'], obj_ale_ssdmbntv2_q['precision'], label=f'SSD_MNV2_Q: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv3small)
plt.plot(obj_ale_ssdmbntv3small['recall'], obj_ale_ssdmbntv3small['precision'], label=f'SSD_MNV3_SMALL: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv3small["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv3large)
plt.plot(obj_ale_ssdmbntv3large['recall'], obj_ale_ssdmbntv3large['precision'], label=f'SSD_MNV3_LARGE: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv3large["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdlitembntv2_voc)
plt.plot(obj_ale_ssdlitembntv2_voc['recall'], obj_ale_ssdlitembntv2_voc['precision'], label=f'SSDLITE_MNV2_VOC: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(obj_ale_ssdmbntv1_voc)
plt.plot(obj_ale_ssdmbntv1_voc['recall'], obj_ale_ssdmbntv1_voc['precision'], label=f'SSD_MNV1_VOC: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')


plt.gcf().suptitle('PATCH:YOLOV2, dataset = INRIA')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_bunch_vs_single_patch/yolov2/pr_curve_yolov2_patch_inria.eps')
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_bunch_vs_single_patch/yolov2/pr_curve_yolov2_patch_inria.png')

















# # OBJ PAPER
# plt.figure()
#
# obj_paper_yolov2 = bb.stat.pr(patch_results_obj_paper_yolov2, annotations, threshold=0.5)
# obj_paper_yolov3 = bb.stat.pr(patch_results_obj_paper_yolov3, annotations, threshold=0.5) #!!!!!!!!!!
# obj_paper_yolov3_tiny = bb.stat.pr(patch_results_obj_paper_yolov3_tiny, annotations, threshold=0.5) #!!!!!!!!!!
# obj_paper_ssdmbntv1 = bb.stat.pr(patch_results_obj_paper_ssdmbntv1, annotations, threshold=0.5)
# obj_paper_ssdmbntv2 = bb.stat.pr(patch_results_obj_paper_ssdmbntv2, annotations, threshold=0.5)
# obj_paper_ssdlitembntv2 = bb.stat.pr(patch_results_obj_paper_ssdlitembntv2, annotations, threshold=0.5)
# obj_paper_ssdmbntv3small = bb.stat.pr(patch_results_obj_paper_ssdmbntv3small, annotations, threshold=0.5)
# obj_paper_ssdmbntv3large = bb.stat.pr(patch_results_obj_paper_ssdmbntv3large, annotations, threshold=0.5)
#
# ''' Plot stuffs '''
#
# plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
#
# ap = bb.stat.ap(obj_paper_yolov2)
# plt.plot(obj_paper_yolov2['recall'], obj_paper_yolov2['precision'], label=f'YOLOV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_yolov2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_yolov3)
# plt.plot(obj_paper_yolov3['recall'], obj_paper_yolov3['precision'], label=f'YOLOV3: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_yolov3["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_yolov3_tiny)
# plt.plot(obj_paper_yolov3_tiny['recall'], obj_paper_yolov3_tiny['precision'], label=f'YOLOV3-TINY: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_yolov3_tiny["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_ssdmbntv1)
# plt.plot(obj_paper_ssdmbntv1['recall'], obj_paper_ssdmbntv1['precision'], label=f'SSD_MNV1: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_ssdmbntv1["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_ssdmbntv2)
# plt.plot(obj_paper_ssdmbntv2['recall'], obj_paper_ssdmbntv2['precision'], label=f'SSD_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_ssdmbntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_ale_ssdlitembntv2)
# plt.plot(obj_ale_ssdlitembntv2['recall'], obj_ale_ssdlitembntv2['precision'], label=f'SSDLITE_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_ssdmbntv3small)
# plt.plot(obj_paper_ssdmbntv3small['recall'], obj_paper_ssdmbntv3small['precision'], label=f'SSD_MNV3_SMALL: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_ssdmbntv3small["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(obj_paper_ssdmbntv3large)
# plt.plot(obj_paper_ssdmbntv3large['recall'], obj_paper_ssdmbntv3large['precision'], label=f'SSD_MNV3_LARGE: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_paper_ssdmbntv3large["recall"].iloc[-1]*100, 2)}%')
#
# plt.gcf().suptitle('PR-curve OBJ_PAPER, ref. = inria_anno, dataset = INRIA')
# plt.gca().set_ylabel('Precision')
# plt.gca().set_xlabel('Recall')
# plt.gca().set_xlim([0, 1.05])
# plt.gca().set_ylim([0, 1.05])
# plt.gca().legend(loc=4)
# plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_comparison/pr_curve_obj_paper_ref_real_inria.eps')
# plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_comparison/pr_curve_obj_paper_ref_real_inria.png')





