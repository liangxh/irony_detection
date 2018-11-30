# -*- coding: utf-8 -*-
from __future__ import print_function
import os

POSITIVE = 'positive'
NEGATIVE = 'negative'


class LexiconLiu05(object):
    """
    Bing Liu, Minqing Hu, and Junsheng Cheng. 2005.
    Opinion observer: Analyzing and comparing opinions on the web.
    In Proceedings of the 14th International World Wide Web conference (WWW-2005).

    https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
    """
    folder = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_data', 'lexicon_liu05')
    paths = {
        POSITIVE: os.path.join(folder, 'positive-words.txt'),
        NEGATIVE: os.path.join(folder, 'negative-words.txt')
    }
    lexicons = dict()

    @classmethod
    def get_lexicon(cls, category):
        if category not in cls.lexicons:
            words = list()
            with open(cls.paths[category]) as file_obj:
                for line in file_obj:
                    if line.startswith(';'):
                        continue
                    line = line.strip()
                    if line == '':
                        continue
                    words.append(line)
            cls.lexicons[category] = set(words)
        return cls.lexicons[category]

    @classmethod
    def get_positive_lexicons(cls):
        return cls.get_lexicon(POSITIVE)

    @classmethod
    def get_negative_lexicons(cls):
        return cls.get_lexicon(NEGATIVE)


if __name__ == '__main__':
    lexicons = LexiconLiu05.get_positive_lexicons()
    print(len(lexicons))

    lexicons = LexiconLiu05.get_negative_lexicons()
    print(len(lexicons))
