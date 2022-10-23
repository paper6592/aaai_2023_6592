from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import copy
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification, EqOddsPostprocessing
import random
from aif360.datasets import BinaryLabelDataset

def splitDataset(train_dataset, train_ratio=0.9, dataset='adult'):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
        favorable_label = 1
        unfavorable_label = 0
    train_dataset = train_dataset.convert_to_dataframe()[0]
    subgroupIndices = {
        '00':[],
        '01':[],
        '10':[],
        '11':[]
    }
    for i in list(train_dataset.index):
        if train_dataset.loc[i][sensitive] == 0 and train_dataset.loc[i][label] == 0: 
            subgroupIndices['00'].append(i)
        elif train_dataset.loc[i][sensitive] == 0 and train_dataset.loc[i][label] == 1: 
            subgroupIndices['01'].append(i)
        elif train_dataset.loc[i][sensitive] == 1 and train_dataset.loc[i][label] == 0: 
            subgroupIndices['10'].append(i)
        elif train_dataset.loc[i][sensitive] == 1 and train_dataset.loc[i][label] == 1: 
            subgroupIndices['11'].append(i)
    trainIndices = []
    valIndices = []
    for i in subgroupIndices:
        random.shuffle(subgroupIndices[i])
        trainIndices.extend(subgroupIndices[i][:int(len(subgroupIndices[i])*train_ratio)])
        valIndices.extend(subgroupIndices[i][int(len(subgroupIndices[i])*train_ratio):])
    trainDataset = copy.deepcopy(train_dataset).loc[trainIndices]
    valDataset = copy.deepcopy(train_dataset).loc[valIndices]
    trainDataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=trainDataset,
                                label_names=[label],
                                protected_attribute_names=[sensitive],
                                unprivileged_protected_attributes=unprivileged_groups)
    valDataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=valDataset,
                                label_names=[label],
                                protected_attribute_names=[sensitive],
                                unprivileged_protected_attributes=unprivileged_groups)
    return trainDataset, valDataset

def train_aif_post_proc(train_dataset, base_classifier='lr', dataset='adult', algorithm='reject'):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
        favorable_label = 1
        unfavorable_label = 0
    # Train on a part of training set
    new_train_set, new_val_set = splitDataset(train_dataset, train_ratio=0.8, dataset=dataset)
    if base_classifier == 'lr':
        base_classifier = LogisticRegression()
    elif base_classifier == 'svm':
        base_classifier = SVC(probability=True)
    base_classifier.fit(new_train_set.features[:,:-1], new_train_set.labels)
    preds_val = base_classifier.predict(new_val_set.features[:,:-1])
    scores_val = base_classifier.predict_proba(new_val_set.features[:,:-1])
    pred_val_set = copy.deepcopy(new_val_set)
    pred_val_set.labels = preds_val
    pred_val_set.scores = scores_val[:,1]
    if algorithm == 'eq':
        postProc = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    elif algorithm == 'cal_eq':
        postProc = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    elif algorithm == 'reject':
        postProc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, metric_name="Average odds difference")
    postProc.fit(new_val_set, pred_val_set)
    return base_classifier, postProc