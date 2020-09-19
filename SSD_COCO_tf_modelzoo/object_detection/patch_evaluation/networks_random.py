import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# clean results on inria

annotations = bb.io.load('anno_darknet', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/test_images/inria_test_annotations_darknet/', class_label_map={0: 'person'})

#clean_results_yolov2 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/yolov2_pytorch/json_files/', class_label_map={0: 'person'})
clean_results_yolov3 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/yolov3_pytorch/json_files/patch_results_random.json', class_label_map={0: 'person'})
# clean_results_yolov4 = bb.io.load('det_coco', 'C:/Users/Alessandro/Desktop/GitLab/YOLOv4_d2p_adv_COCO/json_files/patch_results_random.json', class_label_map={0: 'person'})
#
# clean_results_ssdmbntv1 = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenet_v1_coco_2017_11_17/patch_results_random.json', class_label_map={0: 'person'})
# clean_results_ssdmbntv1_voc = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/json_files/ssdmbntv1/patch_results_random.json", class_label_map={0: 'person'})
# clean_results_ssdmbntv2 = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenet_v2_coco_2018_03_29/patch_results_random.json", class_label_map={0: 'person'})
# clean_results_ssdlitembntv2 = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssdlite_mobilenet_v2_coco_2018_05_09/patch_results_random.json', class_label_map={0: 'person'})
# clean_results_ssdlitembntv2_voc = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/ssd_models_pytorch_haoi/json_files/ssdlitembntv2/patch_results_random.json", class_label_map={0: 'person'})
# clean_results_ssdmbntv1_q = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv1_coco_quantized/patch_results_random.json", class_label_map={0: 'person'})
# clean_results_ssdmbntv2_q = bb.io.load('det_coco', "C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv2_coco_quantized/patch_results_random.json", class_label_map={0: 'person'})
#
# clean_results_ssdmbntv3small = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv3_small/patch_results_random.json', class_label_map={0: 'person'})
# clean_results_ssdmbntv3large = bb.io.load('det_coco', 'C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/json_files/ssd_mobilenetv3_large/patch_results_random.json', class_label_map={0: 'person'})

plt.figure()

#yolov2 = bb.stat.pr(clean_results_yolov2, annotations, threshold=0.5)
yolov3 = bb.stat.pr(clean_results_yolov3, annotations, threshold=0.5)
# yolov4 = bb.stat.pr(clean_results_yolov4, annotations, threshold=0.5)
#
# ssdmbntv1 = bb.stat.pr(clean_results_ssdmbntv1, annotations, threshold=0.5)
# ssdmbntv1_voc = bb.stat.pr(clean_results_ssdmbntv1_voc, annotations, threshold=0.5)
# ssdmbntv2 = bb.stat.pr(clean_results_ssdmbntv2, annotations, threshold=0.5)
# ssdlitembntv2 = bb.stat.pr(clean_results_ssdlitembntv2, annotations, threshold=0.5)
# ssdlitembntv2_voc = bb.stat.pr(clean_results_ssdlitembntv2_voc, annotations, threshold=0.5)
# ssdmbntv1_q = bb.stat.pr(clean_results_ssdmbntv1_q, annotations, threshold=0.5)
# ssdmbntv2_q = bb.stat.pr(clean_results_ssdmbntv2_q, annotations, threshold=0.5)
#
# ssdmbntv3small = bb.stat.pr(clean_results_ssdmbntv3small, annotations, threshold=0.5)
# ssdmbntv3large = bb.stat.pr(clean_results_ssdmbntv3large, annotations, threshold=0.5)

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

# ap = bb.stat.ap(yolov2)
# plt.plot(yolov2['recall'], yolov2['precision'], label=f'YOLOV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov2["recall"].iloc[-1]*100, 2)}%')

ap = bb.stat.ap(yolov3)
plt.plot(yolov3['recall'], yolov3['precision'])#, label=f'YOLOV3: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov3["recall"].iloc[-1]*100, 2)}%')

# ap = bb.stat.ap(yolov4)
# plt.plot(yolov4['recall'], yolov4['precision'], label=f'YOLOV4: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_yolov3_tiny["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv1)
# plt.plot(ssdmbntv1['recall'], ssdmbntv1['precision'], label=f'SSD_MNV1: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv1["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv1_voc)
# plt.plot(ssdmbntv1_voc['recall'], ssdmbntv1_voc['precision'], label=f'SSD_MNV1_VOC: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv1["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv2)
# plt.plot(ssdmbntv2['recall'], ssdmbntv2['precision'], label=f'SSD_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdlitembntv2)
# plt.plot(ssdlitembntv2['recall'], ssdlitembntv2['precision'], label=f'SSDLITE_MNV2: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdlitembntv2_voc)
# plt.plot(ssdlitembntv2_voc['recall'], ssdlitembntv2_voc['precision'], label=f'SSDLITE_MNV2_VOC: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv1_q)
# plt.plot(ssdmbntv1_q['recall'], ssdmbntv1_q['precision'], label=f'SSD_MNV1_Q: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv2_q)
# plt.plot(ssdmbntv2_q['recall'], ssdmbntv2_q['precision'], label=f'SSD_MNV2_Q: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdlitembntv2["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv3small)
# plt.plot(ssdmbntv3small['recall'], ssdmbntv3small['precision'], label=f'SSD_MNV3_SMALL: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv3small["recall"].iloc[-1]*100, 2)}%')
#
# ap = bb.stat.ap(ssdmbntv3large)
# plt.plot(ssdmbntv3large['recall'], ssdmbntv3large['precision'], label=f'SSD_MNV3_LARGE: AP: {round(ap*100, 2)}%') #, RECALL: {round(obj_ale_ssdmbntv3large["recall"].iloc[-1]*100, 2)}%')

#plt.gcf().suptitle('CLEAN_RES, dataset = INRIA')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_bunch_vs_single_patch/pr_curve_rndres_ref_inria.eps', dpi=500)
plt.savefig('C:/Users/Alessandro/PycharmProjects/tf_model_garden/research/object_detection/pr_curves/networks_bunch_vs_single_patch/pr_curve_rndres_ref_inria.png',dpi=500)









