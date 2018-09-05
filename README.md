
python -m scripts.semeval2014_task9 build_text
./scripts/affective_tweets_feature.sh -i data/semeval2014_task9_B/train.txt.text -o data/semeval2014_task9_B/train.lexicon_feat.txt
./scripts/affective_tweets_feature.sh -i data/semeval2014_task9_B/dev.txt.text -o data/semeval2014_task9_B/dev.lexicon_feat.txt
./scripts/affective_tweets_feature.sh -i data/semeval2014_task9_B/test.txt.text -o data/semeval2014_task9_B/test.lexicon_feat.txt

python -m scripts.semeval2018_task1 build_text
./scripts/affective_tweets_feature.sh -i data/semeval2018_task1/E-c-En-train.txt.text -o data/semeval2018_task1/train.lexicon_feat.text
./scripts/affective_tweets_feature.sh -i data/semeval2018_task1/E-c-En-dev.txt.text -o data/semeval2018_task1/dev.lexicon_feat.text


python -m scripts.semeval2018_task3 build_text
./scripts/affective_tweets_feature.sh -i data/semeval2018_task3/train.text -o data/semeval2018_task3/train.lexicon_feat.txt
./scripts/affective_tweets_feature.sh -i data/semeval2018_task3/test.text -o data/semeval2018_task3/test.lexicon_feat.txt
