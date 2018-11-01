# -*- coding: utf-8 -*-
from sklearn.decomposition import TruncatedSVD as SVD


def svd(train, test, n_components):
    svd = SVD(n_components=n_components)
    new_train = svd.fit_transform(X=train)
    new_test = svd.transform(X=test)
    return new_train, new_test

