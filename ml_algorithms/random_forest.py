# -*- coding: utf-8 -*-
# for mac
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import metrics
from sklearn.metrics import auc
from ml_algorithms.ml_algorithm_interface import AlgorithmInterface


class RandomForest(AlgorithmInterface):
    def __init__(self):
        super(RandomForest, self).__init__()

    def feature_engineering(self):
        self.convert_symbolic_feature_into_continuous()

    def train_phase(self):
        random_forest = RandomForestClassifier()

        # The number of trees in the forest
        n_estimators = [100, 500, 900, 1100, 1500]

        # Maximum depth of each tree
        max_depth = [10, 15, 20, 25]

        # hyper parameter tuning
        hyper_parameter_grid = {'n_estimators': n_estimators,
                                'max_depth': max_depth}

        # Set up the random search with 4-fold cross validation
        self.classifier = RandomizedSearchCV(estimator=random_forest,
                                       param_distributions=hyper_parameter_grid,
                                       cv=4, n_iter=20,
                                       scoring='roc_auc',
                                       n_jobs=-1, verbose=2,
                                       return_train_score=True,
                                       random_state=42)

        # Fit on the training data
        self.classifier.fit(self.train_data, self.train_label)

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
