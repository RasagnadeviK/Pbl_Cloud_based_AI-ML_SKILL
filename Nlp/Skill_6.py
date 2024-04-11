import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")
from sklearn.feature_extraction.text import CountVectorizer
simple_train=['Call you tonight','Call me a cab',"Please call me... PLEASE!"]
v=CountVectorizer()
v.fit(simple_train)
v.get_feature_names_out()
print(v.get_feature_names_out())
simple_train_dtm=v.transform(simple_train)
print(simple_train_dtm)
simple_train_dtm.toarray()
print(simple_train_dtm.toarray())
pd.DataFrame(simple_train_dtm.toarray(), columns=v.get_feature_names_out())
print(pd.DataFrame(simple_train_dtm.toarray(), columns=v.get_feature_names_out()))
print(type(simple_train_dtm))
print(simple_train_dtm)
simple_test = ["please don't call me"]
simple_test_dtm=v.transform(simple_test)
simple_test_dtm.toarray()
print(simple_test_dtm.toarray())
pd.DataFrame(simple_test_dtm.toarray(), columns=v.get_feature_names_out())
print(pd.DataFrame(simple_test_dtm.toarray(), columns=v.get_feature_names_out()))
sms = pd.read_csv(r"C:\Users\sampa\OneDrive\Desktop\COLLEGE-CSIT\Sem6\PBL-4\Skill\Data\spam.csv", encoding='latin-1')

sms.dropna(how="any", inplace=True, axis=1)
sms.columns = ['label', 'message']
sms.head()
print(sms.head())
sms.describe()
print(sms.describe())
sms.groupby('label').describe()
print(sms.groupby('label').describe())
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
sms.head()
sms['message_len'] = sms.message.apply(len)
sms.head()
plt.figure(figsize=(12, 8))

sms[sms.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue',
                                       label='Ham messages', alpha=0.6)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red',
                                       label='Spam messages', alpha=0.6)
plt.legend()
plt.xlabel("Message Length")
plt.show()

sms[sms.label=='ham'].describe()
sms[sms.label=='spam'].describe()
sms[sms.message_len == 910].message.iloc[0]
print(sms[sms.message_len == 910].message.iloc[0])
import string
from nltk.corpus import stopwords


def text_process(mess):
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
print(sms.head())
sms['clean_msg'] = sms.message.apply(text_process)
sms.head()
print(sms.head())
type(stopwords.words('english'))
from collections import Counter

words = sms[sms.label == 'ham'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
ham_words = Counter()

for msg in words:
    ham_words.update(msg)

print(ham_words.most_common(50))
words = sms[sms.label == 'spam'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])
spam_words = Counter()

for msg in words:
    spam_words.update(msg)

print(spam_words.most_common(50))
from sklearn.model_selection import train_test_split

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.clean_msg
y = sms.label_num
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm = vect.fit_transform(X_train)
print(type(X_train_dtm), X_train_dtm.shape)
X_test_dtm = vect.transform(X_test)
print(type(X_test_dtm), X_test_dtm.shape)
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
from sklearn import metrics
y_pred_class = nb.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))
metrics.confusion_matrix(y_test, y_pred_class)
print(X_test[y_pred_class > y_test])
print(X_test[y_pred_class < y_test])
print(X_test[4949])
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)
print(metrics.roc_auc_score(y_test, y_pred_prob))
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfid', TfidfTransformer()),
                 ('model', MultinomialNB())])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.confusion_matrix(y_test, y_pred_class))
print(metrics.roc_auc_score(y_test, y_pred_prob))
vect = CountVectorizer(ngram_range=(1, 2))
vect = CountVectorizer(max_df=0.5)
vect = CountVectorizer(min_df=2)