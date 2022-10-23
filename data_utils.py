from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.compose import make_column_selector as selector
import random
import pandas as pd
import copy

# https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline_column_transformer.html
def preprocessDataset(train_path='/media/data_dump/anonymous/aaai_2023_data/adult/raw/realDatasets/imbalance_1', test_path='/media/data_dump/anonymous/aaai_2023_data/adult/raw/original_test.csv', dataset='adult'):
    dataset_train = pd.read_csv(train_path)
    dataset_test = pd.read_csv(test_path)
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        dropAttrs = ['income', 'sex', 'race']
        labelReplacementDict = {'<=50K':0, '>50K':1}
        sensitiveReplacementDict = {'Female':0, 'Male':1}
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        dropAttrs = ['age', 'deposit']
        labelReplacementDict = {'no':0, 'yes':1}
        def sensitiveReplacementFunction(x):
            if x <= 25:
                return 0
            else:
                return 1
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        dropAttrs = ['age','sex', 'foreign_worker', 'credit']
        labelReplacementDict = {'bad':0, 'good':1}
        sensitiveReplacementDict = {'female':0, 'male':1}
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        dropAttrs = ['race', 'category', 'sex']
        labelReplacementDict = {'Survived':0, 'Recidivated':1}
        sensitiveReplacementDict = {'African-American':0, 'Caucasian':1}
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        dropAttrs = ['sensitive', 'label']
        labelReplacementDict = {0:0, 1:1}
        sensitiveReplacementDict = {0:0, 1:1}
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
        favorable_label = 1
        unfavorable_label = 0
    # Separate out labels and sensitive Attributes
    y_train = dataset_train[label]
    z_train = dataset_train[sensitive]
    x_train = dataset_train.drop(columns=dropAttrs)
    y_test = dataset_test[label]
    z_test = dataset_test[sensitive]
    x_test = dataset_test.drop(columns=dropAttrs)
    # Binarize label and sensitive attribute
    y_train.replace(labelReplacementDict, inplace=True)
    y_test.replace(labelReplacementDict, inplace=True)
    if dataset != 'bank':
        z_train.replace(sensitiveReplacementDict, inplace=True)
        z_test.replace(sensitiveReplacementDict, inplace=True)
    else:
        z_train = z_train.apply(sensitiveReplacementFunction)
        z_test = z_test.apply(sensitiveReplacementFunction)
    # Preprocess Data
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(x_train)
    categorical_columns = categorical_columns_selector(x_train)
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()
    if dataset != 'synthetic':
        preprocessor = ColumnTransformer([
            ('one-hot-encoder', categorical_preprocessor, categorical_columns),
            ('standard_scaler', numerical_preprocessor, numerical_columns)]).fit(x_train)
    else:
        preprocessor = ColumnTransformer([
            ('standard_scaler', numerical_preprocessor, numerical_columns)]).fit(x_train)
    tempTrain = preprocessor.transform(x_train)
    tempTest = preprocessor.transform(x_test)
    # Sometimes we get a sparse CSR matrix, sometimes we get a ndarray
    if isinstance(tempTrain, np.ndarray):
        preprocessed_x_train = pd.DataFrame(tempTrain.tolist())
        preprocessed_x_test = pd.DataFrame(tempTest.tolist())
    else:
        preprocessed_x_train = pd.DataFrame(tempTrain.toarray())
        preprocessed_x_test = pd.DataFrame(tempTest.toarray())
    preprocessed_x_train[label] = y_train
    preprocessed_x_test[label] = y_test
    preprocessed_x_train[sensitive] = z_train
    preprocessed_x_test[sensitive] = z_test
    # Convert to AIF format
    binary_train_dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=preprocessed_x_train,
                                label_names=[label],
                                protected_attribute_names=[sensitive],
                                unprivileged_protected_attributes=unprivileged_groups)
    binary_test_dataset = BinaryLabelDataset(favorable_label=favorable_label,
                                unfavorable_label=unfavorable_label,
                                df=preprocessed_x_test,
                                label_names=[label],
                                protected_attribute_names=[sensitive],
                                unprivileged_protected_attributes=unprivileged_groups)
    return binary_train_dataset, binary_test_dataset

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
