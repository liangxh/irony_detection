# -*- coding: utf-8 -*-
import sys
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
    annotate={'hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'},
    all_caps_tag='wrap',
    fix_html=True,
    segmenter='twitter_2018',
    corrector='twitter_2018',
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)


for filename_input in sys.argv[1].split(','):
    filename_output = '{}.ek_2'.format(filename_input)
    with open(filename_input, 'r') as fin, open(filename_output, 'w') as fout:
        for line in fin:
            fout.write(' '.join(text_processor.pre_process_doc(line.strip())) + '\n')
