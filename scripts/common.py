# -*- coding: utf-8 -*-
import commandr
import json
import re
from textblob import TextBlob
from dataset import semeval2014, semeval2018
from dataset.common.const import *


@commandr.command
def pos():
    for config in [
            semeval2014.task9.config, semeval2018.task1.config, semeval2018.task3.config]:
        for mode in [TRAIN, TEST]:
            out_path = config.path(mode, POS)
            text_path = config.path(mode, EK)
            with open(text_path, 'r') as text_obj, open(out_path, 'w') as out_obj:
                for line in text_obj:
                    text = line.strip()
                    if text == '':
                        continue
                    text = re.sub('\<[^\>]+\>', '', text)
                    text = re.sub('\s+', ' ', text)
                    blob = TextBlob(text.decode('utf8'))
                    try:
                        tags = blob.tags
                    except UnicodeDecodeError:
                        print tags
                        print out_path
                    out_obj.write(json.dumps(tags) + '\n')


if __name__ == '__main__':
    commandr.Run()
