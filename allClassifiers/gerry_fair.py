from aif360.algorithms.inprocessing import GerryFairClassifier

def train_gerry_fair(train_dataset):
    inProc = GerryFairClassifier()
    inProc.fit(train_dataset)
    return inProc