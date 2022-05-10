from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from matplotlib import pyplot


import numpy as np

from models import AspectOutput
from modules.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class MebeAspectDTModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 7

    def _represent(self, inputs):
        """

        :param list of models.Input inputs:
        :return:
        """

        features = []
        for ip in inputs:
            _features = [1 if v in ip.text else 0 for v in self.vocab]
            features.append(_features)

        print(len(features[0]))
        return features

    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs)
        ys = [np.array([output.scores[i] for output in outputs]) for i in range(self.NUM_OF_ASPECTS - 1)]
        # self.visualData(ys)
        # yn = []
        # print(X[0].shape)
        # print(len(ys[0]))
        # print(X.shape)


        oversample = SMOTE(sampling_strategy = 0.7, random_state=14)
        undersample = RandomUnderSampler(sampling_strategy = 0.8, random_state=14)

        steps = [('o', oversample), ('u', undersample)]
        pipeline = Pipeline(steps=steps)
        # for i in range(self.NUM_OF_ASPECTS):
        #     X, ys[i] = oversample.fit_resample(X, ys[i])
        # for i in range(self.NUM_OF_ASPECTS):
        #     sum = 0
        #     for output in outputs:
        #         sum += output.scores[i]
        #     print(sum)
        # print(ys[1][1000])

        for i in range(self.NUM_OF_ASPECTS - 1):
            self.models[i].fit(X, ys[i])

    #     self.visualizeData(ys, yn)
    #     self.visualData(yn)
    # def save(self, path):
    #     pass
    #
    # def load(self, path):
    #     pass
    #
    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs)

        outputs = []
        predicts = [self.models[i].predict(X) for i in range(self.NUM_OF_ASPECTS - 1)]
        for ps in zip(*predicts):
            labels = list(range(self.NUM_OF_ASPECTS))
            scores = list(ps)
            if 1 in scores:
                scores.append(0)
            else:
                scores.append(1)
            outputs.append(AspectOutput(labels, scores))

        return outputs
    #
    # def visualizeData(self, ys, yn):
    #     aspects = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    #     value1 = []
    #     value0 = []
    #     for i in range(self.NUM_OF_ASPECTS):
    #         value0.append(len(ys[i]))
    #         value1.append(len(yn[i]))
    #     print(len(value0))
    #     print(len(value1))
    #     indx = np.arange(len(aspects))
    #     bar_width = 0.35
    #     score_label = np.arange(0, 110, 10)
    #     fig, ax = pyplot.subplots()
    #     bar0 = ax.bar(indx - bar_width / 2, value0, bar_width, label='before')
    #     bar1 = ax.bar(indx + bar_width / 2, value1, bar_width, label='after')
    #     ax.set_xticks(indx)
    #     ax.set_xticklabels(aspects)
    #     ax.set_yticks(score_label)
    #     ax.set_yticklabels(score_label)
    #     ax.legend()
    #
    #     def insert_data_labels(bars):
    #         for bar in bars:
    #             bar_height = bar.get_height()
    #             ax.annotate('{0:.0f}'.format(bar_height),
    #                         xy=(bar.get_x() + bar.get_width() / 2, bar_height),
    #                         xytext=(0, 3),
    #                         textcoords='offset points',
    #                         ha='center',
    #                         va='bottom'
    #                         )
    #
    #     insert_data_labels(bar0)
    #     insert_data_labels(bar1)
    #     pyplot.show()
    #
    # def visualData(self, ys):
    #     aspects = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    #     value1 = []
    #     value0 = []
    #     for i in range(self.NUM_OF_ASPECTS):
    #         counter = Counter(ys[i])
    #         for k, v in counter.items():
    #             per = v / len(ys[i]) * 100
    #             # print('Aspect%d, n=%d (%.3f%%)' % (k, v, per))
    #             if k == 0:
    #                 value0.append(round(per, 3))
    #             if k == 1:
    #                 value1.append(round(per, 3))
    #
    #     indx = np.arange(len(aspects))
    #     bar_width = 0.35
    #     score_label = np.arange(0, 110, 10)
    #     fig, ax = pyplot.subplots()
    #     bar0 = ax.bar(indx - bar_width / 2, value0, bar_width, label='value 0')
    #     bar1 = ax.bar(indx + bar_width / 2, value1, bar_width, label='value 1')
    #     ax.set_xticks(indx)
    #     ax.set_xticklabels(aspects)
    #     ax.set_yticks(score_label)
    #     ax.set_yticklabels(score_label)
    #     ax.legend()
    #
    #     def insert_data_labels(bars):
    #         for bar in bars:
    #             bar_height = bar.get_height()
    #             ax.annotate('{0:.0f}'.format(bar_height),
    #                         xy=(bar.get_x() + bar.get_width() / 2, bar_height),
    #                         xytext=(0, 3),
    #                         textcoords='offset points',
    #                         ha='center',
    #                         va='bottom'
    #                         )
    #
    #     insert_data_labels(bar0)
    #     insert_data_labels(bar1)
    #     pyplot.show()