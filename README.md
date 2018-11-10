#!/bin/bash

python -m scripts.semeval2014_task9 build_text_label
prefix=data/semeval2014_task9_B
for mode in {train,test}; do
    ./scripts/affective_tweets_feature.sh -i ${prefix}/${mode}.text -o ${prefix}/${mode}.lexicon_feat
done

python -m scripts.semeval2018_task1 build_text_label
prefix=data/semeval2018_task1
for mode in {train,test}; do
    ./scripts/affective_tweets_feature.sh -i ${prefix}/${mode}.text -o ${prefix}/${mode}.lexicon_feat
done

python -m scripts.semeval2018_task3 build_text_label
prefix=data/semeval2018_task3
for mode in {train,test}; do
    ./scripts/affective_tweets_feature.sh -i ${prefix}/${mode}.text -o ${prefix}/${mode}.lexicon_feat
done

'''
https://www.receptiviti.com/liwc-api-get-started/
'''
