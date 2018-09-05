#! /bin/bash
set -e

while getopts "i:o:" arg; do
    case $arg in
    i)
        INPUT=${OPTARG}
    ;;
    o)
        OUTPUT=${OPTARG}
    ;;
    esac
done

DIR_LAB=${HOME}/lab
PATH_TO_TEXT=${INPUT}
PATH_TO_TEXT_ARFF=${INPUT}.tmp.arff
PATH_TO_VEC_ARFF={INPUT}_vec.tmp.arff
PATH_TO_VEC=${OUTPUT}

WEKA_JAR=${DIR_LAB}/weka/weka/weka.jar

echo 'generating .arff for text..'

cd ${DIR_LAB}/irony_detection
python -m scripts.process_arff text -i ${PATH_TO_TEXT} -o ${PATH_TO_TEXT_ARFF}


echo 'building lexicon feature vector...'

java -Xmx4G -cp ${WEKA_JAR} weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector  \
	-i ${PATH_TO_TEXT_ARFF} -o ${PATH_TO_VEC_ARFF} \
	-stemmer weka.core.stemmers.NullStemmer \
	-stopwords-handler "weka.core.stopwords.Null " \
	-I 1 -U -tokenizer "weka.core.tokenizers.TweetNLPTokenizer " \
	-A -D -F -H -J -N -P -Q -R -T

echo 'transforming .arff of lexicon feature to customized format'

python -m scripts.process_arff vec -i ${PATH_TO_VEC_ARFF} -o ${PATH_TO_VEC}

echo 'remove temporary files'

rm ${PATH_TO_TEXT_ARFF}
rm ${PATH_TO_VEC_ARFF}
