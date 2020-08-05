import numpy as np

from collections import defaultdict

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


class CVModel:

    def __init__(self, get_model, fit_model):
        """ 
        Runs double cross validation with input models.
    
        Parameters:
        get_model - function that initializes CV with specific model 
        fit_model - function that runs CV with specific model 
        """
        self.get_model = get_model
        self.fit_model = fit_model
        self.models = None
        
    def fit(self, X, y, n_splits, seed=19, metric='auc', verbose=False):
        """ Performs cross-validation.

            Returns:
            average folds score for a defined metric
        """
        kfold = KFold(n_splits, random_state=seed, shuffle=True)
        splits = kfold.split(X, y)
            
        self.models = []
        scores = defaultdict(list)
        
        for fold, (train_ids, val_ids) in enumerate(splits):
            
            train_X, train_y = X.iloc[train_ids], y.iloc[train_ids]
            test_X, test_y = X.iloc[val_ids], y.iloc[val_ids]
            
            model = self.get_model
            score = self.fit_model(model, train_X, train_y, test_X, test_y)
            
            scores['auc'].append(score)
            self.models.append(model)
            
            if verbose:
                print(f"fold {fold + 1:>2}, score: {score:.4f}")
            
        return np.mean(scores['auc'])
    
    def predict(self, X):
        """ Returns:
            average output for LIST of models
        """
        return np.mean([model.predict_proba(X)[:, 1] for model in self.models], axis=0)
    
    def predict_stack(self, X):
        """ Returns:
            prediction output for LIST of LISTS of models 
        """
        return np.mean([model.predict(X) for model in self.models], axis=0)
    

def get_lightgbm(**model_params):
    return LGBMClassifier(**model_params)


def fit_lightgbm(model, train_X, train_y, test_X, test_y):
    model.fit(
        train_X, train_y, 
        eval_set=(test_X, test_y),
        early_stopping_rounds=50, verbose=False, eval_metric='auc', 
    )
    
    return model.best_score_['valid_0']['auc']


def get_cvmodel(**model_params):
    return CVModel(get_lightgbm(**model_params), fit_lightgbm)


def fit_cvmodel(model, train_X, train_y, test_X, test_y):
    model.fit(train_X, train_y, n_splits=5)
    score = roc_auc_score(test_y.values, model.predict(test_X))
    return score