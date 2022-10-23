import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import argparse
from allClassifiers import jiang_nachum, reweighing, exp_grad, aif_post_proc, fairgan, meta_fair, adv_deb, prej_remover, gerry_fair, lfr
from tqdm import tqdm as tqdm
from data_utils import splitDataset, preprocessDataset
from utils import test_preproc, test_inproc, test_postproc
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--base_classifier", type=str,
                    help="Base Classifier to use", default='lr')
parser.add_argument("--dataset", type=str,
                    help="Dataset to run on", default='adult')
parser.add_argument("--algorithm", type=str,
                    help="Algorithm to use", default='jiang_nachum')
parser.add_argument("--constraint", type=str, 
                    help='Fairness Constraint (used only when required)', default='eop')
#parser.add_argument("--gpu", type=str, default="4")
#parser.add_argument("--experimentName", type=str, default='')
#parser.add_argument("--resumeRun", type=int, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    # Underrepresentation Beta_Pos and Beta_Neg results
    # First index of the dictionary will always have the AIF metric object, further indices might have algorithm specific metrics
    results = {}
    results['undersample'] = {}
    print('Performing Under-Representation Experiments')
    for i in tqdm(os.listdir(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/betaDatasets/')): 
        train_dataset, test_dataset_biased = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/test_betaDatasets/{i}', args.dataset)
        _, test_dataset_original = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/original_test.csv', args.dataset)
        _, test_dataset_balanced = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/balanced/balanced_test.csv', args.dataset)
        test_datasets = {
            'balanced':test_dataset_balanced,
            'biased':test_dataset_biased,
            'original':test_dataset_original
        }
        if args.algorithm == 'base':
            model = LogisticRegression().fit(train_dataset.features[:,:-1], train_dataset.labels)
            results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'jiang_nachum':
            model = jiang_nachum.train_jiang_nachum_reweighing(train_dataset, modelType=args.base_classifier, constraint=args.constraint)
            results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'rew':
            model = reweighing.train_rew(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
            results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'lfr':
            try:
                model = lfr.train_lfr(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
            except ValueError as ve:
                print(ve)
                results['undersample'][i] = None
        elif args.algorithm == 'fairgan':
            try:
                model = fairgan.train_fairgan_data_classifier(train_dataset, base_classifier=args.base_classifier)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
            except ValueError as ve:
                print(ve)
                results['undersample'][i] = None
        elif args.algorithm == 'adv_deb':
            # Changed Adv Deb AIF code to train a LR instead
            model, sess = adv_deb.train_adv_deb(train_dataset, args.dataset)
            results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            sess.close()
        elif args.algorithm == 'exp_grad':
            model = exp_grad.train_exp_grad(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
            results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'prej_remover':
            model = prej_remover.train_prej_remover(train_dataset, dataset=args.dataset)
            results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'gerry_fair':
            model = gerry_fair.train_gerry_fair(train_dataset)
            results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'meta_fair':
            try:
                # No base classifier required for Meta Fair
                model = meta_fair.train_meta_fair(train_dataset, constraint=args.constraint, dataset=args.dataset)
                results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            except Exception as e:
                print(e)
                results['undersample'][i] = None
        elif args.algorithm == 'reject':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='reject')
            results['undersample'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
        elif args.algorithm == 'eq':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='eq')
            results['undersample'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
        elif args.algorithm == 'cal_eq':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='cal_eq')
            results['undersample'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
    
    # Label Bias Results
    results['label_bias'] = {}
    print()
    print('Performing Label Bias Experiments')
    for i in tqdm(os.listdir(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/labelBiasDatasets_train/')): 
        train_dataset, test_dataset_biased = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/labelBiasDatasets_train/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/labelBiasDatasets_test/{i}', args.dataset)
        _, test_dataset_original = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/labelBiasDatasets_train/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/original_test.csv', args.dataset)
        _, test_dataset_balanced = preprocessDataset(f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/raw/labelBiasDatasets_train/{i}', f'/media/data_dump/anonymous/aaai_2023_data/{args.dataset}/balanced/balanced_test.csv', args.dataset)
        test_datasets = {
            'balanced':test_dataset_balanced,
            'biased':test_dataset_biased,
            'original':test_dataset_original
        }
        if args.algorithm == 'base':
            model = LogisticRegression().fit(train_dataset.features[:,:-1], train_dataset.labels)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'jiang_nachum':
            model = jiang_nachum.train_jiang_nachum_reweighing(train_dataset, modelType=args.base_classifier, constraint=args.constraint)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'rew':
            model = reweighing.train_rew(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'lfr':
            try:
                model = lfr.train_lfr(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
            except ValueError as ve:
                print(ve)
                results['label_bias'][i] = None
        elif args.algorithm == 'fairgan':
            model = fairgan.train_fairgan_data_classifier(train_dataset, base_classifier=args.base_classifier)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'adv_deb':
            # Changed Adv Deb AIF code to train a LR instead
            model, sess = adv_deb.train_adv_deb(train_dataset, args.dataset)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
            sess.close()
        elif args.algorithm == 'exp_grad':
            model = exp_grad.train_exp_grad(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'gerry_fair':
            model = gerry_fair.train_gerry_fair(train_dataset)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'prej_remover':
            model = prej_remover.train_prej_remover(train_dataset, dataset=args.dataset)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'meta_fair':
            try:
                # No base classifier required for Meta Fair
                model = meta_fair.train_meta_fair(train_dataset, constraint=args.constraint, dataset=args.dataset)
                results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
            except Exception as e:
                print(e)
                results['label_bias'][i] = None
        elif args.algorithm == 'reject':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='reject')
            results['label_bias'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
        elif args.algorithm == 'eq':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='eq')
            results['label_bias'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
        elif args.algorithm == 'cal_eq':
            base_model, model = aif_post_proc.train_aif_post_proc(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset, algorithm='cal_eq')
            results['label_bias'][i] = test_postproc(base_model, model, test_datasets, args.dataset)
    # Store Results
    with open(f'/media/data_dump/anonymous/aaai_2023_data/results/{args.algorithm}__{args.constraint}__{args.dataset}__{args.base_classifier}.pkl', 'wb') as f:
        pickle.dump(results, f)
