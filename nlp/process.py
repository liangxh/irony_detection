# -*- coding: utf-8 -*-
import re
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


def naive_tokenize(text):
    text = re.sub('@\S+', 't_user', text)
    text = re.sub('https?://\S+', ' t_url ', text)
    tokens = tokenizer.tokenize(text)
    tokens = map(lambda t: t.lower(), tokens)
    tokens = map(lambda t: t[1:] if len(t) > 1 and t.startswith('#') else t, tokens)
    return tokens
