# -*- coding: utf-8 -*-
import json
from dataset.common.const import *
from dataset.semeval2015_task11.config import config

NOT_AVAILABLE = 'NOT AVAILABLE'


class Processor(object):
    @classmethod
    def load_tid_score(cls, mode):
        """
        :param mode: string
        :return: list<pair<tweet_id, score>>
        """
        if mode == TRAIN:
            path = config.path_train_id_score
        elif mode == TEST:
            path = config.path_trial_id_score
        else:
            raise ValueError('invalid mode: {}'.format(mode))
        tid_score = list()
        with open(path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(',')
                tid = parts[0]
                score = float(parts[1])
                tid_score.append((tid, score))
        return tid_score

    @classmethod
    def load_download_tweets(cls, mode):
        if mode == TRAIN:
            path = config.path_train_raw
        elif mode == TEST:
            path = config.path_trial_raw
        elif mode == 'new':
            path = config.path_new_raw
        else:
            raise ValueError('invalid mode: {}'.format(mode))
        return json.load(open(path, 'r'))

    @classmethod
    def load_new_mapping(cls):
        """
        :return:
            tid_score, dict<tweet_id, score>
            tid_mapping, dict<old_tweet_id, new_tweet_id>
        """
        path = config.path_new_id_score
        tid_score = dict()
        tid_mapping = dict()
        with open(path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split(',')
                new_tid = parts[0]
                old_tid = parts[1]
                score = float(parts[2])
                tid_score[new_tid] = score
                tid_mapping[old_tid] = new_tid
        return tid_score, tid_mapping
