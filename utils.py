import copy
from aif360.metrics import ClassificationMetric
import numpy as np

def test_preproc(model, test_datasets, dataset):
    # Requires model to have a predict function
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
    results = {}
    for i in test_datasets:
        new_preds_test = model.predict(test_datasets[i].features[:,:-1])
        pred_test_set = copy.deepcopy(test_datasets[i])
        pred_test_set.labels = new_preds_test
        results[i] = ClassificationMetric(test_datasets[i], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    return results

def test_inproc(model, test_datasets, dataset):
    # Requires model to have a predict function
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
    results = {}
    for i in test_datasets:
        pred_test_set = model.predict(test_datasets[i])
        results[i] = ClassificationMetric(test_datasets[i], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    return results

def test_postproc(base_model, model, test_datasets, dataset='adult'):
    # Requires model to have a predict function
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
    results = {}
    for i in test_datasets:
        test_datasets[i].scores = base_model.predict_proba(test_datasets[i].features[:, :-1])[:,1]
        pred_test_set = model.predict(test_datasets[i])
        results[i] = ClassificationMetric(test_datasets[i], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    return results

def test_zafar(model, test_datasets, dataset='adult'):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
    results = {}
    for i in test_datasets:
        predicted_labels = np.sign(np.dot(test_datasets[i].features[:,:-1], model))
        predicted_labels[predicted_labels == -1] = 0
        pred_test_set = copy.deepcopy(test_datasets[i])
        pred_test_set.labels = predicted_labels
        results[i] = ClassificationMetric(test_datasets[i], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    return results