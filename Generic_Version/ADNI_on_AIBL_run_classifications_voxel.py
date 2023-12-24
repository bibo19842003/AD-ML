import os
from os import path

import pandas as pd
import numpy as np
import nibabel as nib

from clinica.pipelines.machine_learning.input import CAPSVoxelBasedInput
import clinica.pipelines.machine_learning.ml_utils as utils
import clinica.pipelines.machine_learning.voxel_based_io as vbio
from sklearn.metrics import roc_auc_score


# ================== config begin ==================
# predict caps
caps_dir = '/mnt/4t1/homework/ei/data_less3/OUTPUT/ADNI/CAPS'

# train caps folder
adni_caps_dir = '/mnt/4t1/homework/all_data/boss1_data/AD_CN_caps_maching'

# classification model path
adni_output_dir = '/mnt/4t1/homework/ei/DATA/all_put/ADNI/OUTPUT'

# output classification
output_dir = '/mnt/4t1/homework/ei/DATA/all_put/ADNI/CLASSIFICATION/classification_test1'

group_id = 'reg'

image_types = ['T1w']

# predict lists_by_task folder
tasks_dir = '/mnt/4t1/homework/ei/data_less3/OUTPUT/ADNI/TSV'

# predict subjects_sessions.tsv
predict_subjects_sessions_tsv = "list_T1_ADNI.tsv"

# predict diagnoses.tsv
predict_diagnoses_tsv = "diagnosis_36_ADNI.tsv"

# train lists_by_task folder
adni_tasks_dir = '/mnt/4t1/homework/all_data/boss1_data/tsv'

# train subjects_sessions.tsv
train_subjects_sessions_tsv = "subject_ad_cn.tsv"

# train diagnoses.tsv
train_diagnoses_tsv = "participants_diagnosis.tsv"


tasks = [('CN', 'AD')]

# smoothing
fwhm = 8

pvc = None
# ================== config end ==================

##### Voxel based classifications ######


for image_type in image_types:
    for task in tasks:
        subjects_visits_tsv = path.join(tasks_dir, predict_subjects_sessions_tsv)
        diagnoses_tsv = path.join(tasks_dir, predict_diagnoses_tsv)

        adni_subjects_visits_tsv = path.join(adni_tasks_dir, train_subjects_sessions_tsv)
        adni_diagnoses_tsv = path.join(adni_tasks_dir, train_diagnoses_tsv)

        classification_dir = path.join(output_dir, image_type, 'voxel_based', 'linear_svm', '%s_vs_%s' % (task[0], task[1]))
        adni_classifier_dir = path.join(adni_output_dir, image_type, 'voxel_based', 'linear_svm', '%s_vs_%s' % (task[0], task[1]), 'classifier')

        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

        print("Running %s" % classification_dir)

        #adni_images = CAPSVoxelBasedInput(adni_caps_dir, adni_subjects_visits_tsv, adni_diagnoses_tsv, group_id, image_type, fwhm, modulated='on', pvc=pvc, mask_zeros=False)
        adni_images = CAPSVoxelBasedInput({"caps_directory":adni_caps_dir, 
                                           "subjects_visits_tsv":adni_subjects_visits_tsv, 
                                           "diagnoses_tsv":adni_diagnoses_tsv, 
                                           "group_label":group_id, 
                                           "image_type":image_type, 
                                           "fwhm":fwhm, 
                                           "modulated":'on', 
                                           "use_pvc_data":pvc, 
                                           "mask_zeros":False})

        input_images = CAPSVoxelBasedInput({"caps_directory":caps_dir, 
                                            "subjects_visits_tsv":subjects_visits_tsv, 
                                            "diagnoses_tsv":diagnoses_tsv, 
                                            "group_label":group_id,
                                            "image_type":image_type, 
                                            "fwhm":fwhm, 
                                            "modulated":'on', 
                                            "use_pvc_data":pvc, 
                                            "mask_zeros":False})

        adni_x, adni_orig_shape, adni_data_mask = vbio.load_data(adni_images.get_images(), mask=True)

        weights = np.loadtxt(path.join(adni_classifier_dir, 'weights.txt'))
        w = vbio.revert_mask(weights, adni_data_mask, adni_orig_shape).flatten()

        b = np.loadtxt(path.join(adni_classifier_dir, 'intersect.txt'))

        x = input_images.get_x()
        y = input_images.get_y()

        y_hat = np.dot(w, x.transpose()) + b

        y_binary = (y_hat > 0) * 1.0

        evaluation = utils.evaluate_prediction(y, y_binary)

        auc = roc_auc_score(y, y_hat)
        evaluation['AUC'] = auc

        print(evaluation)

        del evaluation['confusion_matrix']

        res_df = pd.DataFrame(evaluation, index=['i', ])
        res_df.to_csv(path.join(classification_dir, 'results_auc.tsv'), sep='\t')
