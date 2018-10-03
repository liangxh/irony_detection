# -*- coding: utf-8 -*-
import sys
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
    annotate={'hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'},
    fix_html=True,
    segmenter='twitter',
    corrector='twitter',
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)


filename_input = sys.argv[1]
filename_output = filename_input + '.out'

with open(filename_input, 'r') as fin, open(filename_output, 'w') as fout:
    for line in fin:
        fout.write(' '.join(text_processor.pre_process_doc(s)) + '\n')
