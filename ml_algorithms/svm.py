# -*- coding: utf-8 -*-
# for mac
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import auc
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from ml_algorithms.ml_algorithm_interface import AlgorithmInterface


class SVMAlgorithm(AlgorithmInterface):
    def __init__(self):
        super(SVMAlgorithm, self).__init__()

    def feature_engineering(self):
        self.convert_symbolic_feature_into_continuous()

    def train_phase(self):
        pipe_svc = Pipeline([('scl', StandardScaler()),
                             ('clf', SVC(random_state=1))])
        # param_range = [10 ** c for c in range(-4, 4)]
        param_range = [0.0001, 0.001]
        hyper_parameter_grid = {
            'clf__C': param_range,
            'clf__gamma': param_range,
            'clf__kernel': ['linear', 'rbf']
        }

        # Set up the random search with 4-fold cross validation
        self.classifier = RandomizedSearchCV(estimator=pipe_svc,
                                             param_distributions=hyper_parameter_grid,
                                             cv=4, n_iter=5,
                                             scoring='roc_auc',
                                             n_jobs=-1, verbose=2,
                                             return_train_score=True,
                                             random_state=42)

        # Fit on the training data
        self.classifier.fit(self.train_data, self.train_label)
        print("123")

    def test_phase(self):
        y_predict = self.classifier.predict(self.test_data)
        print("accuracy: %f" % accuracy_score(self.test_label, y_predict))
        print("precision: %f" % precision_score(self.test_label, y_predict, average="macro"))
        print("recall: %f" % recall_score(self.test_label, y_predict, average="macro"))

        fpr, tpr, thresholds = metrics.roc_curve(y_predict, self.test_label)
        plt.plot(fpr, tpr, marker='o')
        plt.show()
        auc_score = auc(fpr, tpr)
        print("AUC: %f" % auc_score)
