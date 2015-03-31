__author__ = 'newuser'

categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
     categories=categories, shuffle=True, random_state=42)
print twenty_train.target_names
# print(" ".join(twenty_train.data[0].split("\n")[:3]))
# for t in twenty_train.target[:10]:
#     print twenty_train.target_names[t]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
                    ])
text_clf=text_clf.fit(twenty_train.data,twenty_train.target)

import numpy as ny
twenty_test=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)
doc_test=twenty_test.data
predicted=text_clf.predict(doc_test)
print(ny.mean(predicted==twenty_test.target))
