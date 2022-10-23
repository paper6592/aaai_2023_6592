from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.metrics import ClassificationMetric
import os
from aif360.algorithms.preprocessing import OptimPreproc, Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult, get_distortion_compas
from aif360.algorithms.inprocessing import GerryFairClassifier, MetaFairClassifier, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification, EqOddsPostprocessing
import copy
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle
from sklearn.svm import SVC
from data_utils import preprocessDataset, splitDataset


def trainClassifiers(train_dataset, test_dataset, dataset='adult', classifier='LR'):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
        optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
        }
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
        optim_options = {
        "distortion_fun": get_distortion_compas,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
        }
    results={}
    # Baseline
    results['base'] = {}
    if classifier == 'LR':
        base_classifier = LogisticRegression()
    elif classifier == 'SVM':
        base_classifier = SVC(probability=True)
    base_classifier.fit(train_dataset.features[:,:-1], train_dataset.labels)
    pred_test_set = copy.deepcopy(test_dataset)
    pred_test_set.labels = base_classifier.predict(pred_test_set.features[:,:-1])
    results['base']['lr'] = ClassificationMetric(test_dataset, pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Preprocessing Algorithms
    results['pre'] = {}
    # Skipping Disparate Impact remover since it requires the conditional distributions to be same between training and testing sets
    #dis_impact_remover = DisparateImpactRemover(sensitive_attribute='sex')
    #transformed_train_dataset = dis_impact_remover.fit_transform(train_dataset)
    #transformed_test_dataset = dis_impact_remover.fit_transform(test_dataset)
    #for i in ['rew', 'optPre']:
    for i in ['rew']:
        if i == 'rew':
            preProc = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        #elif i == 'optPre':
        #    preProc = OptimPreproc(OptTools, optim_options, unprivileged_groups, privileged_groups)
        #preProc.fit(train_dataset)
        preProc.fit(train_dataset)
        new_train_dataset = preProc.transform(train_dataset)
        if classifier == 'LR':
            cls = LogisticRegression()
        elif classifier == 'SVM':
            cls = SVC(probability=True)
        cls.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels, sample_weight=new_train_dataset.instance_weights)
        new_preds_test = cls.predict(test_dataset.features[:,:-1])
        pred_test_set = copy.deepcopy(test_dataset)
        pred_test_set.labels = new_preds_test
        results['pre'][i] = ClassificationMetric(test_dataset, pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Inprocessing Algorithms
    results['in'] = {}
    for i in ['prej_remover', 'exp_grad_dp', 'exp_grad_eo']:
        #if i == 'gerry_dp':
        #    inProc = GerryFairClassifier(fairness_def='SP')
        #if i == 'gerry_eo':
        #    if classifier == 'LR':
        #        inProc = GerryFairClassifier(fairness_def='FP', predictor=LogisticRegression())
        #    elif classifier == 'SVM':
        #        inProc = GerryFairClassifier(fairness_def='FP', predictor=SVC(probability=True))
        #elif i == 'meta_fair_dp':
        #    inProc = MetaFairClassifier(sensitive_attr=sensitive, type='sr')
        #elif i == 'meta_fair_eo':
        #    inProc = MetaFairClassifier(sensitive_attr=sensitive, type='fdr')
        if i == 'prej_remover':
            inProc = PrejudiceRemover(sensitive_attr=sensitive, class_attr=label)
        elif i == 'exp_grad_dp':
            if classifier == 'LR':
                inProc = ExponentiatedGradientReduction(LogisticRegression(), constraints='DemographicParity')
            elif classifier == 'SVM':
                inProc = ExponentiatedGradientReduction(SVC(probability=True), constraints='DemographicParity')
        elif i == 'exp_grad_eo':
            if classifier == 'LR':
                inProc = ExponentiatedGradientReduction(LogisticRegression(), constraints='EqualizedOdds')
            elif classifier == 'SVM':
                inProc = ExponentiatedGradientReduction(SVC(probability=True), constraints='EqualizedOdds')
        inProc.fit(train_dataset)
        pred_test_set = inProc.predict(test_dataset)
        results['in'][i] = ClassificationMetric(test_dataset, pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Postprocessing Algorithms
    results['post'] = {}
    new_train_set, new_val_set = splitDataset(train_dataset, train_ratio=0.8, dataset=dataset)
    if classifier == 'LR':
        base_classifier = LogisticRegression()
    elif classifier == 'SVM':
        base_classifier = SVC(probability=True)
    base_classifier.fit(new_train_set.features[:,:-1], new_train_set.labels)
    preds_val = base_classifier.predict(new_val_set.features[:,:-1])
    scores_val = base_classifier.predict_proba(new_val_set.features[:,:-1])
    pred_val_set = copy.deepcopy(new_val_set)
    pred_val_set.labels = preds_val
    pred_val_set.scores = scores_val[:,1]
    test_dataset.scores = base_classifier.predict_proba(test_dataset.features[:, :-1])[:,1]
    for i in ['eq', 'cal_eq', 'reject']:
        if i == 'eq':
            postProc = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        elif i == 'cal_eq':
            postProc = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        elif i == 'reject':
            postProc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, metric_name="Average odds difference")
        postProc.fit(new_val_set, pred_val_set)
        pred_test_set = postProc.predict(test_dataset)
        results['post'][i] = ClassificationMetric(test_dataset, pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # End comment
    return results

def trainClassifiers_multipleTestSets(train_dataset, test_datasets, dataset='adult', classifier='LR'):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
        optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
        }
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
        optim_options = {
        "distortion_fun": get_distortion_compas,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
        }
    results={}
    for i in test_datasets:
        results[i] = {}
    # Baseline
    for i in results:
        results[i]['base'] = {}
    if classifier == 'LR':
        base_classifier = LogisticRegression()
    elif classifier == 'SVM':
        base_classifier = SVC(probability=True)
    base_classifier.fit(train_dataset.features[:,:-1], train_dataset.labels)
    # Compute base classifier results on all test sets
    for i in results:
        pred_test_set = copy.deepcopy(test_datasets[i])
        pred_test_set.labels = base_classifier.predict(pred_test_set.features[:,:-1])
        results[i]['base']['lr'] = ClassificationMetric(test_datasets[i], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Preprocessing Algorithms
    for i in results:
        results[i]['pre'] = {}
    # Skipping Disparate Impact remover since it requires the conditional distributions to be same between training and testing sets
    #dis_impact_remover = DisparateImpactRemover(sensitive_attribute='sex')
    #transformed_train_dataset = dis_impact_remover.fit_transform(train_dataset)
    #transformed_test_dataset = dis_impact_remover.fit_transform(test_dataset)
    #for i in ['rew', 'optPre']:
    for i in ['rew']:
        if i == 'rew':
            preProc = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        #elif i == 'optPre':
        #    preProc = OptimPreproc(OptTools, optim_options, unprivileged_groups, privileged_groups)
        #preProc.fit(train_dataset)
        new_train_dataset = preProc.transform(train_dataset)
        if classifier == 'LR':
            cls = LogisticRegression()
        elif classifier == 'SVM':
            cls = SVC(probability=True)
        cls.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels, sample_weight=new_train_dataset.instance_weights)
        # For all test sets
        for j in results:
            new_preds_test = cls.predict(test_datasets[j].features[:,:-1])
            pred_test_set = copy.deepcopy(test_datasets[j])
            pred_test_set.labels = new_preds_test
            results[j]['pre'][i] = ClassificationMetric(test_datasets[j], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Inprocessing Algorithms
    for i in results:
        results[i]['in'] = {}
    for i in ['prej_remover', 'exp_grad_dp', 'exp_grad_eo']:
        #if i == 'gerry_dp':
        #    inProc = GerryFairClassifier(fairness_def='SP')
        #if i == 'gerry_eo':
        #    if classifier == 'LR':
        #        inProc = GerryFairClassifier(fairness_def='FP', predictor=LogisticRegression())
        #    elif classifier == 'SVM':
        #        inProc = GerryFairClassifier(fairness_def='FP', predictor=SVC(probability=True))
        #elif i == 'meta_fair_dp':
        #    inProc = MetaFairClassifier(sensitive_attr=sensitive, type='sr')
        #elif i == 'meta_fair_eo':
        #    inProc = MetaFairClassifier(sensitive_attr=sensitive, type='fdr')
        if i == 'prej_remover':
            inProc = PrejudiceRemover(sensitive_attr=sensitive, class_attr=label)
        elif i == 'exp_grad_dp':
            if classifier == 'LR':
                inProc = ExponentiatedGradientReduction(LogisticRegression(), constraints='DemographicParity')
            elif classifier == 'SVM':
                inProc = ExponentiatedGradientReduction(SVC(probability=True), constraints='DemographicParity')
        elif i == 'exp_grad_eo':
            if classifier == 'LR':
                inProc = ExponentiatedGradientReduction(LogisticRegression(), constraints='EqualizedOdds')
            elif classifier == 'SVM':
                inProc = ExponentiatedGradientReduction(SVC(probability=True), constraints='EqualizedOdds')
        inProc.fit(train_dataset)
        for j in results:
            pred_test_set = inProc.predict(test_datasets[j])
            results[j]['in'][i] = ClassificationMetric(test_datasets[j], pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # Postprocessing Algorithms
    for i in results:
        results[i]['post'] = {}
    new_train_set, new_val_set = splitDataset(train_dataset, train_ratio=0.8, dataset=dataset)
    if classifier == 'LR':
        base_classifier = LogisticRegression()
    elif classifier == 'SVM':
        base_classifier = SVC(probability=True)
    base_classifier.fit(new_train_set.features[:,:-1], new_train_set.labels)
    preds_val = base_classifier.predict(new_val_set.features[:,:-1])
    scores_val = base_classifier.predict_proba(new_val_set.features[:,:-1])
    pred_val_set = copy.deepcopy(new_val_set)
    pred_val_set.labels = preds_val
    pred_val_set.scores = scores_val[:,1]
    for j in results:
        test_dataset = copy.deepcopy(test_datasets[j])
        test_dataset.scores = base_classifier.predict_proba(test_dataset.features[:, :-1])[:,1]
        for i in ['eq', 'cal_eq', 'reject']:
            if i == 'eq':
                postProc = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            elif i == 'cal_eq':
                postProc = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            elif i == 'reject':
                postProc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, metric_name="Average odds difference")
            postProc.fit(new_val_set, pred_val_set)
            pred_test_set = postProc.predict(test_dataset)
            results[j]['post'][i] = ClassificationMetric(test_dataset, pred_test_set, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # End comment
    return results

dataset='bank'
baseClassifier = 'SVM'
allDatasetResults = {}
for i in tqdm(os.listdir(f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/betaDatasets/')):
    train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/original_test.csv', dataset)
    _, test_dataset_biased = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/test_betaDatasets/{i}', dataset)
    _, test_dataset_balanced = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{dataset}/balanced/balanced_test.csv', dataset)
    test_datasets = {
        'balanced':test_dataset_balanced,
        'biased':test_dataset_biased,
        'original':test_dataset_original
    }
    try:
        allDatasetResults[i] = trainClassifiers_multipleTestSets(train_dataset, test_datasets, dataset, classifier=baseClassifier)
    except:
        print(i)
        raise

with open(f'/media/data_dump/anonymous/aaai_2023_data/results/{dataset}_{baseClassifier}_multipleTestSets_results.pkl', 'wb') as f:
    pickle.dump(allDatasetResults, f)
