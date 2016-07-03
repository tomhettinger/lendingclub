# -*- coding: utf-8 -*-
"""
Plotting
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_training_deviance(clf, X_test, y_test, nsteps):
    """Plot training deviance of a model."""
    # compute test set deviance
    test_score = np.zeros((nsteps,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    
    plt.figure(figsize=(12, 6))
    plt.title('Deviance')
    plt.plot(np.arange(nsteps) + 1, clf.train_score_, 'b-', label='Training Set Deviance')
    plt.plot(np.arange(nsteps) + 1, test_score, 'r-', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


def plot_feature_importance(clf, feature_names, ndisplay=20):
    """Plot feature importance for a model."""
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[-ndisplay:]    # only use the top 20 features
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance (Top %d)' % ndisplay)
    plt.savefig('feature_importance.png')
    #plt.show()
    
    
def plot_coef(clf, feature_names, ndisplay=30):
    feature_tup = zip(feature_names, clf.coef_)
    feature_tup.sort(key=lambda tup: abs(tup[1]))
    feature_tup = feature_tup[-ndisplay:]
    names, importance = zip(*feature_tup)
    
    pos = np.arange(len(names)) + .5
    plt.figure(figsize=(12, ndisplay/3.))
    plt.barh(pos, importance, align='center')
    plt.yticks(pos, names)
    plt.xlabel('Coef')
    plt.title('Variable Importance (Top %d)' % ndisplay)
    plt.savefig('lm_coeff.png')
    #plt.show()
    
    
def plot_2dhist(x, y, bins=(50, 50)):
    plt.hist2d(x, y, cmap=pl.cm.jet)