from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.model_selection import GridSearchCV

from gensim import corpora, models
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation

import jieba
import jieba.posseg as pseg
import jieba.analyse

import io
import re
import codecs
import numpy as np
import pandas as pd


class datapp:
    #def __init__(self):
    #    return None

    def stopfile(self, f):
        stopword = set()
        with codecs.open(f, "r", "utf-8") as fd:
            for line in fd.readlines():
                #print "stop",line.strip(),type(line.strip())
                stopword.add(line.strip())
        return stopword

    def w2v(self, f):
        w2vec = []
        with codecs.open(f, "r", "utf-8") as fd:
            for line in fd.readlines():
                w2vec.append(line.strip())
        return w2vec
    
    def dp_pp2(self, f, feature):
        d = {"label": [], "content": []}
        #stopword = self.stopfile("stopword.txt")
        #w2vec = self.w2v("feature_lstm.txt")
        with codecs.open(f, "r", "utf-8") as fd:
            for line in fd.readlines():
                l = line.strip().split()
                if len(l) == 1:
                    continue
                if l[0] == u"利空":
                    d["label"].append(0)
                else:
                    d["label"].append(1)
                #cut and remove stopword
                l[1] = re.sub(r'\.|\,|[0-9]+%|[0-9]|[a-z]|[A-Z]', '', l[1])
                words = set(jieba.cut(l[1]))
                new_words = [i for i in words if i in feature]

                d["content"].append(new_words)

        #df = pd.DataFrame(data=d)
        return np.array(d["content"]), np.array(d["label"])

    def dp_pp(self, f, feature):
        d = {"label": [], "content": []}
        #stopword = self.stopfile("stopword.txt")
        w2vec = self.w2v("w2vec.txt")
        with codecs.open(f, "r", "utf-8") as fd:
            for line in fd.readlines():
                l = line.strip().split()
                if len(l) == 1:
                    continue
                if l[0] == u"利空":
                    d["label"].append(0)
                else:
                    d["label"].append(1)
                #cut and remove stopword
                l[1] = re.sub(r'\.|\,|[0-9]+%|[0-9]|[a-z]|[A-Z]', '', l[1])
                words = set(jieba.cut(l[1]))
                new_words = [i for i in words if i in w2vec]

                d["content"].append(new_words)

        #df = pd.DataFrame(data=d)
        return np.array(d["content"]), np.array(d["label"])

    def preproces(self, f):
        d = {"label": [], "content": []}
        stopword = self.stopfile("stopword.txt")
        with codecs.open(f, "r", "utf-8") as fd:
            for line in fd.readlines():
                l = line.strip().split()
                if len(l) == 1:
                    continue
                d["label"].append(l[0])
                #cut and remove stopword
                l[1] = re.sub(r'\.|\,|[0-9]+%|[0-9]|[a-z]|[A-Z]', '', l[1])
                words = set(jieba.cut(l[1]))

                new_words = ' '.join(
                    [i for i in words if (i not in stopword) and (len(i) > 1)])

                d["content"].append(new_words)

        df = pd.DataFrame(data=d)
        return df  

    def vectorize(self, data):
        tfidf = TFIDF()
        fit_t = tfidf.fit_transform(data["content"])
        weight = pd.DataFrame(fit_t.toarray())

        word = tfidf.get_feature_names()
        #print weight.shape
        return word, weight, data["label"].values

    def vectorize2(self, f, feature):
        data = self.preprocess(f)
        tfidf = TFIDF(vocabulary=feature)
        fit_t = tfidf.fit_transform(data["content"])
        weight = pd.DataFrame(fit_t.toarray())

        return weight.values, data["label"].values

    def chosen_feature(self):
        ftre = set()
        with codecs.open('selected_feature.txt', 'r+', 'utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = line.strip('\n').split()
                ftre.add(line[0])
        f.close()
        feature = {b: a for a, b in enumerate(ftre)}
        return feature



def ctoix(np):
    for i,v in enumerate(np): 
        np[i] = [w2indx[n] for n in v if n in w2indx]
    return np


lstm = datapp()
feature = lstm.chosen_feature()
#train: longest 52
x_train,y_train = lstm.dp_pp('cnews.train.txt', feature)
#x_val,y_val = lstm.dp_pp('cnews.val.txt', feature)
#x_test,y_test = lstm.dp_pp('cnews.test.txt', feature)

#word2vec build weight_dictionary

word_vectors = KeyedVectors.load_word2vec_format('sgns.sogounews.bigram-char.txt', binary=False)

dictionary = corpora.Dictionary(x_train)
corpus = [dictionary.doc2bow(text) for text in x_train]

w2indx = {v: k+1 for k, v in dictionary.items()}

w2vec = {word: word_vectors[word] for word in w2indx.keys() if word in word_vectors}
x_val,y_val = lstm.dp_pp('cnews.val.txt', list(w2indx))
x_test,y_test = lstm.dp_pp('cnews.test.txt', list(w2indx))


X_train = ctoix(x_train)
X_val = ctoix(x_val)
X_test = ctoix(x_test)


x_train = sequence.pad_sequences(X_train, maxlen=300)
x_val = sequence.pad_sequences(X_val, maxlen=300)
x_test = sequence.pad_sequences(X_test, maxlen=300)

embedding_weights = np.zeros((len(w2indx)+1, 300))

for word, index in w2indx.items():
        embedding_weights[index, :] = w2vec[word]



#LSTM
def train_lstm(vocab_dim, n_symbols, embeding_weights, input_length, b_size,
               n_epoch, x_train, y_train, x_val, y_val, x_test, y_test):
    his = []
    sc = []
    ac = []
    print("start keras lstm...")
    model = Sequential()
    model.add(
        Embedding(
            output_dim=vocab_dim,
            input_dim=n_symbols,
            #mask_zero=True,
            weights=[embeding_weights],
            input_length=input_length))
    model.add(
        LSTM(
            output_dim=60,
            activation='sigmoid',
            inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    #model.add(Dense(60, activation="relu"))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print("compiling...")
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    hist = model.fit(
        x_train,
        y_train,
        nb_epoch=n_epoch,
        batch_size=b_size,
        validation_data=(x_val, y_val))

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    his.append(hist)
    sc.append(score)
    ac.append(acc)
    return his,sc, ac

result = {}
for i in [5]:
    his, score, acc = train_lstm(vocab_dim, num_symbols, embedding_weights,
                                 input_length, batch_size, i, x_train, y_train,
                                 x_val, y_val, x_test, y_test)
    result[i] = {"his": his, "score": score, "acc": acc}


