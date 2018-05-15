# -*- coding: utf-8 -*-
import re
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


def process_tweet(text):
    text = re.sub('@\S+', 't_user', text)
    text = re.sub('https?://\S+', ' t_url ', text)
    tokens = tokenizer.tokenize(text)
    tokens = map(lambda t: t.lower(), tokens)
    return tokens
