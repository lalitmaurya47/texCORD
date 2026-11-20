import os
import json
import pickle

import cv2
import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, KFold


def load_dataset(dataset):
    if dataset.name == 'eye':
        labels, ims = [], []
        for i, label in enumerate(['Normal', 'VH']):
            base_path = os.path.join(dataset.path, label)

            ims_i = [
                torch.from_numpy(
                    cv2.resize(
                        cv2.imread(
                            os.path.join(base_path, f), cv2.IMREAD_GRAYSCALE
                        ), (512, 512))) for f in os.listdir(base_path)
            ]

            ims += ims_i
            labels += [i] * len(ims_i)

        ims = torch.stack(ims)
        labels = np.array(labels, dtype=np.int32)
        patients = np.arange(len(labels))

    elif dataset.name == 'lungs':
        vids = [f for f in os.listdir(dataset.path) if f.endswith('.npz')]
        ims, blines, covid, severity_score, patient_ids = [], [], [], [], []
        for vid in vids:
            data = np.load(os.path.join(dataset.path, vid))
            n_ims = len(data['ims'])

            ims += [
                torch.from_numpy(im) for im in data['ims']
            ]

            blines += [data['blines']] * n_ims
            covid += [data['covid']] * n_ims
            severity_score += [data['severity_score']] * n_ims
            patient_ids += [data['patient_id']] * n_ims

        ims = torch.stack(ims)
        labels = {'blines': np.hstack(blines), 'covid': np.hstack(covid), 'severity_score': np.hstack(severity_score)}
        patients = np.hstack(patient_ids)

    elif dataset.name == 'lung-inflammation':
        vids = [f for f in os.listdir(dataset.path) if f.endswith('.npz')]
        with open(os.path.join(dataset.path, 'clinical_values.json'), 'r') as f:
            clinical_values = json.load(f)

        ims, labels, patient_ids = [], [], []
        for vid in vids:
            data = np.load(os.path.join(dataset.path, vid))

            metadata = clinical_values[vid.rstrip('.npz')]

            ims.append(data['ims'])
            labels.append([data['covid']] + [metadata[p] for p in dataset.params])
            patient_ids.append(data['patient_id'])

        ims = np.stack(ims)
        labels = np.abs(np.vstack(labels))
        patients = np.hstack(patient_ids)

        tmp = np.zeros_like(patients)
        kf = KFold(n_splits=dataset.n_splits, shuffle=True, random_state=7)

        for group_num, (_, test_idx) in enumerate(kf.split(np.unique(patients))):
            tmp[np.isin(patients, test_idx)] = group_num

        patients = tmp

    elif dataset.name == 'spine-binary':
        ims, labels, patients = [], [], []
        for i, train_test in enumerate(['train', 'test']):
            src = os.path.join(dataset.path, train_test)
            for f in os.listdir(src):
                data = np.load(os.path.join(src, f))
                ims.append(data['im'])
                labels.append(data['label'])
                patients.append(i)

        ims = torch.from_numpy(np.stack(ims))
        labels = np.array(labels)
        patients = np.array(patients)

    elif dataset.name == 'echonet':
        ims, labels, split_idx = [], [], []

        for idx, split in enumerate(['train', 'test']):
            src_dir = os.path.join(dataset.path, split)
            files = os.listdir(src_dir)

            for file in files:
                with open(os.path.join(src_dir, file), 'rb') as f:
                    data = pickle.load(f)

                images = data['images']
                if len(images) >= dataset.k:
                    indices = np.linspace(0, len(images), dataset.k).round().clip(0, len(images)-1).astype(int)

                    ims.append([images[i] for i in indices])
                    labels.append(data['ef'])
                    split_idx.append(idx)

        ims = np.array(ims)
        labels = np.array(labels)
        patients = np.array(split_idx)

    elif dataset.name == 'liver':
        ims, labels, patient_ids = [], [], []

        file = loadmat(dataset.path, simplify_cells=True)

        for patient in file['data']:
            ims.append(patient['images'])
            labels.append(patient['fat'])
            patient_ids.append(patient['id'])

        ims = np.stack(ims)
        labels = np.hstack(labels)
        patients = np.hstack(patient_ids)

        train, test = train_test_split(range(len(np.unique(patients))+1), shuffle=True, test_size=15, random_state=7)

        tmp = np.full_like(patients, -1)
        tmp[np.isin(patients, train)] = 0
        tmp[np.isin(patients, test)] = 1

        patients = tmp

    return ims, labels, patients
