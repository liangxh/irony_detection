# -*- coding: utf-8 -*-
import os


class AFinn(object):
    """
    Finn Arup Nielsen. 2011.
    A new anew: Evaluation of a word list for sentiment analysis in microblogs.
    In Proceedings of the ESWC2011 Workshop on ’Making Sense of Microposts’:
    Big things come in small packages
    (http://arxiv.org/abs/1103.2903).
    """
    data_path = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_data', 'AFINN-111.txt')

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = dict(map(
                lambda (k, v):
                    (k, int(v)),
                    [line.split('\t') for line in open(self.data_path)]
            ))
        return self._data

    def get_score(self, word):
        """
        :param word: string
        :return: float, -5. ~ 5.
        """
        return float(self.data.get(word.lower(), 0.))


afinn = AFinn()

if __name__ == '__main__':
    print afinn.get_score('LOVE')
