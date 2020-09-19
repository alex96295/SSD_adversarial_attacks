#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import brambox as bb
import brambox.io.parser.box as bbb
import matplotlib.pyplot as plt
from IPython.display import display


# In[5]:


annotations = bb.io.load('anno_darknet', "C:/Users/Alessandro/Desktop/GitLab/INRIAPerson_annotations_darknet/Test/labels_padded", class_label_map={0: 'person'})
#annotations = bb.io.load('anno_darknet', "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2k_adv_COCO_thys2019/test_results_mytrial/clean/yolo-labels_padded", class_label_map={0: 'person'})

#annotations = bb.io.load('anno_darknet', "C:/Users/Alessandro/Desktop/GitLab/INRIAPerson_annotations_darknet/Test/labels", class_label_map={0: 'person'})
#annotations = bb.io.load('anno_darknet', "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2k_adv_COCO_thys2019/test_results_mytrial/clean/yolo-labels", class_label_map={0: 'person'})

patch_results_obj = bb.io.load('det_coco', "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2k_adv_COCO_thys2019/json_files/yolov2_patches/patch_obj_results_1attempt_withpadding.json", class_label_map={0: 'person'})
#patch_results_obj = bb.io.load('det_coco', "C:/Users/Alessandro/Desktop/GitLab/YOLOv2_d2k_adv_COCO_thys2019/json_files/yolov2_patches/patch_obj_results.json", class_label_map={0: 'person'})

plt.figure()

obj_only = bb.stat.pr(patch_results_obj, annotations, threshold=0.5)

''' Plot stuffs '''

plt.plot([0, 1.05], [0, 1.05], '--', color='gray')


ap = bb.stat.ap(obj_only)
plt.plot(obj_only['recall'], obj_only['precision'], label=f'OBJ: AP: {round(ap*100, 2)}%')


plt.gcf().suptitle('YOLOV2 vs OBJ')
plt.gca().set_ylabel('Precision')
plt.gca().set_xlabel('Recall')
plt.gca().set_xlim([0, 1.05])
plt.gca().set_ylim([0, 1.05])
plt.gca().legend(loc=4)
plt.savefig("C:/Users/Alessandro/Desktop/padvsnopad/pad_vs_real.png")
plt.savefig("C:/Users/Alessandro/Desktop/padvsnopad/pad_vs_real.eps")
#plt.show()

