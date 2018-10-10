# 流程

* 神经网络训练

python algo/main.py train config.yaml semeval2014_task9
>> OUTPUT_KEY: gru_1539175546

* 利用训练好的模型为另一个数据集生成特征

python algo/main.py feat semeval2014_task9 gru_1539175546 semeval2018_task3
>> OUTPUT_KEY: semeval2014_task9.gru_1539175546

* 将output_key添加到config_svm.yaml中的feat_keys字段, 以此利用多组特征重新SVM分类器

python algo/svm.py main config_svm.yaml semeval2018_task3 A



```
python algo/main.py train semeval2018_task3 A

python algo/main.py train semeval2018_task1 love
python algo/main.py feat semeval2018_task1 love_gru_1539178720

python algo/main.py train semeval2014_task9
python algo/main.py feat semeval2014_task9 gru_1539175546
```

```
python algo/svm.py main semeval2018_task3 A
```