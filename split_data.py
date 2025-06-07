import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

def is_train(row, dropout, train, val, test):
    if row in dropout:
        return 'Dropout'
    elif row in train:
        return 'Train'
    elif row in val:
        return 'Validation'
    else:
        return 'Test'

# Train - Val - Test Ratio: 0.64: 0.16: 0.20
# data_usage_ratio: how many percentage of data we use, and how many we discard
def make_split_tag(patient_list, data_usage_ratio=1, seed=0):
    patient_id = [os.path.basename(i) for i in patient_list]
    
    # Fix test_size -> train_size here
    train_patient, dropout_patient = train_test_split(patient_id,train_size=data_usage_ratio, random_state=seed)
    train_patient, test_patient = train_test_split(train_patient,test_size=0.2, random_state=seed)
    train_patient, val_patient = train_test_split(train_patient,test_size=0.2, random_state=seed)
    data_split = [ is_train(pid, dropout_patient, train_patient, val_patient, test_patient) for pid in patient_id ]
    meta = []
    for p, d in zip(patient_id, data_split):
        meta.append({
            "folder_name": p,
            "data_split": d
        })
    
    return meta

if __name__ == '__main__':
    data_dir = "/media/volume1/BraTS2025/7/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"
    
    meta_path = './meta'
    os.makedirs(meta_path, exist_ok=True)
    
    # set determinism to ensure reproducibility
    seed = 0 
    patient_list = os.listdir(data_dir)
    meta = make_split_tag(patient_list, data_usage_ratio=0.2, seed=seed)
    
    df = pd.DataFrame(meta)
    df.to_csv('./meta/data_split_Usage20.csv', index=False)