# -*- coding: utf-8 -*-
import re
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()


def naive_tokenize(text):
    """
    预处理如下依次执行
    * 将 @XXXX 替换成 t_user
    * 将 http://xxxx 和 https://xxxx 替换成 t_url
    * 进行切词
    * 转换成小写
    * 将长度大于且以#开头的token默认为hashtag, 去除其#

    :param text:
    :return:
    """
    text = re.sub('@\S+', 't_user', text)
    text = re.sub('https?://\S+', ' t_url ', text)
    tokens = tokenizer.tokenize(text)
    tokens = map(lambda t: t.lower(), tokens)
    tokens = map(lambda t: t[1:] if len(t) > 1 and t.startswith('#') else t, tokens)
    return tokens
