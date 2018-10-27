# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from tools.basic import iterate_file


data_path = os.path.join(os.environ['HOME'], 'lab', 'SentiWordNet_3.0.txt')

POS = 'POS'
POS_SCORE = 'PosScore'
NEG_SCORE = 'NegScore'
SYNSET_TERMS = 'SynsetTerms'


class SentiWordNet(object):
    """
    正/负向情感极性评分均为 0 ~ 1
    """

    def __init__(self):
        self._data = None
        self._phrases = None
        self._words = None

    @property
    def data(self):
        if self._data is None:
            self._data = defaultdict(lambda: defaultdict(lambda: {POS_SCORE: 0., NEG_SCORE: 0.}))
            columns = None
            last_line = None
            started = False
            for line in iterate_file(data_path):
                if line.startswith('#'):
                    last_line = line
                    continue

                if not started:
                    started = True
                    columns = last_line[1:].strip().split('\t')
                values = line.split('\t')
                item = dict(zip(columns, values))
                terms = item[SYNSET_TERMS].split(' ')
                words = map(lambda term: term.split('#')[0], terms)

                if item[POS_SCORE] == '0' and item[NEG_SCORE] == '0':
                    continue

                for word in words:
                    _score_dict = self._data[word][item[POS]]
                    self._data[word][item[POS]] = {
                        POS_SCORE: max(float(item[POS_SCORE]), _score_dict[POS_SCORE]),
                        NEG_SCORE: max(float(item[NEG_SCORE]), _score_dict[NEG_SCORE])
                    }
        return self._data

    @property
    def phrases(self):
        if self._phrases is None:
            self._phrases = dict()
            for key, value in self.data.items():
                if '_' in key:
                    max_pos = max(map(lambda _item: _item[POS_SCORE], value.values()))
                    max_neg = max(map(lambda _item: _item[NEG_SCORE], value.values()))

                    phrase = key.replace('_', ' ')
                    self._phrases[phrase] = {POS_SCORE: max_pos, NEG_SCORE: max_neg}
        return self._phrases


senti_wordnet = SentiWordNet()

if __name__ == '__main__':
    print len(senti_wordnet.data)
    print len(senti_wordnet.phrases)
