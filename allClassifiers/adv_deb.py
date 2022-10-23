import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

def train_adv_deb(train_dataset, dataset='adult'):
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
    
    sess = tf.Session()
    inProc = AdversarialDebiasing(unprivileged_groups, privileged_groups, scope_name='temp_1', sess=sess)
    
    inProc.fit(train_dataset)
    return inProc, sess