# -*- coding: utf-8 -*-
import abc
import pandas as pd
from collections import OrderedDict


class AlgorithmInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.columns = OrderedDict()
        self.load_data()
        self.classifier = None

    def load_data(self):
        labels_2_dict = {'normal': 0, 'attack': 1, 'satan': 1, 'smurf': 1, 'spy': 1, 'teardrop': 1,
                         'warezclient': 1, 'warezmaster': 1, 'unknown': 1,
                         'back': 1, 'buffer_overflow': 1, 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1,
                         'ipsweep': 1, 'land': 1, 'loadmodule': 1, 'multihop': 1, 'neptune': 1,
                         'nmap': 1, 'perl': 1, 'phf': 1, 'pod': 1, 'portsweep': 1, 'rootkit': 1, 'mailbomb': 1,
                         'apache2': 1, 'processtable': 1, 'mscan': 1, 'saint': 1, 'httptunnel': 1, 'snmpgetattack': 1,
                         'snmpguess': 1, 'sendmail': 1, 'ps': 1, 'xsnoop': 1, 'named': 1, 'xterm': 1, 'worm': 1,
                         'xlock': 1, 'sqlattack': 1, 'udpstorm': 1}

        columns_name_path = "NSL_KDD/Field Names.csv"
        with open(columns_name_path, "r") as file:
            for line in file:
                key, data_type = line.split(",")
                self.columns[key] = data_type.replace("\n", "")

        train_data_path = "NSL_KDD/KDDTrain+.csv"
        data = pd.read_csv(train_data_path)
        self.train_data = data.iloc[:, 0:-2]
        self.train_label = data.iloc[:, -2]
        self.train_data.columns = self.columns.keys()
        self.train_label.replace(labels_2_dict, inplace=True)

        test_data_path = "NSL_KDD/KDDTest+.csv"
        data = pd.read_csv(test_data_path)
        self.test_data = data.iloc[:, 0:-2]
        self.test_label = data.iloc[:, -2]
        self.test_data.columns = self.columns.keys()
        self.test_label.replace(labels_2_dict, inplace=True)

    def convert_symbolic_feature_into_continuous(self):
        for key in self.columns.keys():
            if self.columns[key] != "symbolic":
                continue

            category = dict()
            for i, symbolic in enumerate(self.train_data[key].unique()):
                category[symbolic] = i

            self.train_data[key].replace(category, inplace=True)
            self.test_data[key].replace(category, inplace=True)
            del category

    def run(self):
        self.feature_engineering()
        self.train_phase()
        self.test_phase()

    @abc.abstractmethod
    def feature_engineering(self):
        pass

    @abc.abstractmethod
    def train_phase(self):
        pass

    @abc.abstractmethod
    def test_phase(self):
        pass
