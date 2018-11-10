# -*- coding: utf-8 -*-
"""
http://sentistrength.wlv.ac.uk/results.php?text=alas&submit=Detect+Sentiment
"""
import os
from tools.basic import iterate_file

BOOSTER_WORD = 'BoosterWordList'
EMOTICON_LOOKUP_TABLE = 'EmoticonLookupTable'
EMOTION_LOOKUP_TABLE = 'EmotionLookupTable'
ENGLISH_WORD = 'EnglishWordList'
IDIOM_LOOKUP_TABLE = 'IdiomLookupTable'
NEGATING_WORDS = 'NegatingWordList'
QUESTION_WORDS = 'QuestionWords'
SLANG_LOOK_TABLE = 'SlangLookupTable'

root_path = os.path.join(os.environ['HOME'], 'lab', 'irony_detection_data', 'SentiStrength_Data')


class SentiStrength(object):
    """
    正向情感评分为  1 ~ 5
    负向情感评分为 -1 ~ -5
    """

    def __init__(self):
        self._words = None
        self._boosters = None
        self._negating_words = None
        self._question_words = None
        self._slang_lookup_table = None
        self._idiom_lookup_table = None
        self._emoticon_lookup_table = None
        self._emotion_lookup_table = None

    @classmethod
    def path(cls, key):
        return os.path.join(root_path, '{}.txt'.format(key))

    @property
    def boosters(self):
        """
        :return: dict(string, float)
        """
        if self._boosters is None:
            self._boosters = dict()
            for line in iterate_file(self.path(BOOSTER_WORD)):
                parts = line.split('\t')
                word, value = parts[:2]
                self._boosters[word] = int(value)
        return self._boosters

    @property
    def words(self):
        if self._words is None:
            self._words = set(iterate_file(self.path(ENGLISH_WORD)))
        return self._words

    @property
    def idiom_lookup_table(self):
        """
        :return: dict(string, float)
        """
        if self._idiom_lookup_table is None:
            self._idiom_lookup_table = dict()
            for line in iterate_file(self.path(IDIOM_LOOKUP_TABLE)):
                parts = line.split('\t')
                word, value = parts[:2]
                self._idiom_lookup_table[word] = int(value)
        return self._idiom_lookup_table

    @property
    def negating_words(self):
        """
        :return: set of string
        """
        if self._negating_words is None:
            self._negating_words = set(iterate_file(self.path(NEGATING_WORDS)))
        return self._negating_words

    @property
    def question_words(self):
        """
        :return: set of string
        """
        if self._question_words is None:
            self._question_words = set(iterate_file(self.path(QUESTION_WORDS)))
        return self._question_words

    @property
    def slang_lookup_table(self):
        """
        :return: dict<string, string>
        """
        if self._slang_lookup_table is None:
            self._slang_lookup_table = dict()
            for line in iterate_file(self.path(SLANG_LOOK_TABLE)):
                word, long_term = line.split('\t')
                self._slang_lookup_table[word] = long_term.replace('\xa0', '')
        return self._slang_lookup_table

    @property
    def emoticon_lookup_table(self):
        """
        :return: dict<string, float>
        """
        if self._emoticon_lookup_table is None:
            self._emoticon_lookup_table = dict()
            for line in iterate_file(self.path(EMOTICON_LOOKUP_TABLE)):
                emoticon, value = line.split('\t')[:2]
                self._emoticon_lookup_table[emoticon] = int(value)
        return self._emoticon_lookup_table

    @property
    def emotion_lookup_table(self):
        """
        :return: dict<string pattern, float>
        """
        if self._emotion_lookup_table is None:
            self._emotion_lookup_table = dict()
            for line in iterate_file(self.path(EMOTION_LOOKUP_TABLE)):
                parts = line.split('\t')
                word, value = parts[:2]
                value = int(value)
                if word in self._emotion_lookup_table and self._emotion_lookup_table[word] != value:
                    raise Exception('crush {}'.format(word))
                self._emotion_lookup_table[word] = value
        return self._emotion_lookup_table


senti_strength = SentiStrength()


if __name__ == '__main__':
    print len(senti_strength.words)
    print senti_strength.boosters
    print senti_strength.negating_words
    print senti_strength.question_words
    print senti_strength.slang_lookup_table
    print senti_strength.idiom_lookup_table
    print len(senti_strength.emoticon_lookup_table)
    print len(senti_strength.emotion_lookup_table)
