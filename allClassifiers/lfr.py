from aif360.algorithms.preprocessing import LFR
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_lfr(train_dataset, base_classifier, dataset):
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
    preProc = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    preProc.fit(train_dataset)
    new_train_dataset = preProc.transform(train_dataset)
    if base_classifier == 'lr':
        model = LogisticRegression()
    elif base_classifier == 'svm':
        model = SVC(probability=True)
    model.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels)
    return model