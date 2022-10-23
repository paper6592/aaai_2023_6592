from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from aif360.algorithms.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction

def train_exp_grad(train_dataset, base_classifier='lr', constraint='dp'):
    if constraint == 'dp':
        constraint = 'DemographicParity'
    elif constraint == 'eodds':
        constraint = 'EqualizedOdds'
    
    if base_classifier == 'lr':
        inProc = ExponentiatedGradientReduction(LogisticRegression(), constraints=constraint)
    elif base_classifier == 'svm':
        inProc = ExponentiatedGradientReduction(SVC(probability=True), constraints=constraint)
    
    inProc.fit(train_dataset)
    return inProc