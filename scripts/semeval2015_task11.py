# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import json
from dataset.common.const import *
from dataset.semeval2015_task11.config import config
from dataset.semeval2015_task11.process import Processor, NOT_AVAILABLE


@commandr.command
def build_text_label():
    new_tid_score, new_tid_mapping = Processor.load_new_mapping()
    new_downloaded_tweets = Processor.load_download_tweets('new')

    # TEST
    downloaded_tweets = dict()
    tid_score = dict()
    for mode in [TRAIN, TEST]:
        downloaded_tweets[mode] = Processor.load_download_tweets(mode)
        tid_score[mode] = Processor.load_tid_score(mode)

    train_tids = set(dict(tid_score[TRAIN]).keys())
    print(len(tid_score[TRAIN]), len(train_tids))

    test_tids = set(dict(tid_score[TEST]).keys())
    print(len(tid_score[TEST]), len(test_tids))

    train_tscore = dict(tid_score[TRAIN])
    test_tscore = dict(tid_score[TEST])

    valid_tids = {
        TRAIN: train_tids - test_tids,
        TEST: test_tids
    }

    for mode in [TRAIN, TEST]:
        n_sample = len(tid_score[mode])
        found = 0
        score_mismatch = 0

        text_path = config.path(mode, TEXT)
        label_path = config.path(mode, LABEL)

        with open(text_path, 'w') as fobj_text, open(label_path, 'w') as fobj_label:
            for tid, score in tid_score[mode]:
                if tid not in valid_tids[mode]:
                    continue

                text = downloaded_tweets[mode].get(tid)

                if text == NOT_AVAILABLE and tid in new_tid_mapping:
                    new_tid = new_tid_mapping[tid]
                    text = new_downloaded_tweets[new_tid]

                    text = text.replace('\\n', '')
                    if text.startswith('RT'):
                        parts = text.split(': ', 1)
                        if len(parts) == 2:
                            text = parts[1]
                        else:
                            print(text)
                            parts = text.split(' ', 2)
                            text = parts[2]

                    new_score = new_tid_score[new_tid]
                    if score != new_score:
                        if tid in train_tscore and new_score == train_tscore[tid]:
                            score = new_score
                        else:
                            print('FUCK')
                            print('{} {}'.format(tid, score))
                            print('{} {}'.format(new_tid, new_score))
                            #raise Exception

                if text == NOT_AVAILABLE:
                    continue

                found += 1
                fobj_text.write('{}\n'.format(text.encode('utf8')))
                fobj_label.write('{}\n'.format(score))
        print('found: {} / {}'.format(found, n_sample))
        print('score mismatch: {}'.format(score_mismatch))


if __name__ == '__main__':
    commandr.Run()
