# -*- coding: utf-8 -*-
# for mac
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
        self.classifier = RandomForestClassifier(n_estimators=50)
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
