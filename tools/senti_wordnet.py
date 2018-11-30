# -*- coding: utf-8 -*-
from __future__ import print_function
from nltk.corpus import sentiwordnet as swn


class SentiWordNet(object):
    """
    正/负向情感极性评分均为 0 ~ 1
    """
    @classmethod
    def get(cls, word, pos=None):
        synsets = swn.senti_synsets(string=word, pos=pos)
        return synsets


senti_wordnet = SentiWordNet()

if __name__ == '__main__':
    print(senti_wordnet.get('happy', 'a'))
