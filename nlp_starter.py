import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
from sklearn.feature_extraction import text 
matplotlib.style.use('ggplot')

def plot_interaction(df, feature1,feature2,target,x_label=None, y_label=None,title=''):
    if x_label is None:
        x_label = feature1
    if y_label is None:
        y_label = feature2
    plt.scatter(df[feature1], df[feature2], c=labels.astype(np.float))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for i in df[target].unique():
        avg1 = df[ df[target]==i ][feature1].mean()
        avg2 = df[ df[target]==i ][feature2].mean()
        plt.gca().text(avg1, avg2, 'c{}'.format(i), style='italic',
                bbox={'facecolor':'tomato', 'alpha':1, 'pad':2})
    plt.show()


def plotPCA(X, labels, n_components=2):
    pca = PCA(n_components=n_components)
    XX = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print ('zachovana variancia pre PCA: ',var.sum())
    plt.scatter(XX[:, 0], XX[:, 1],c=labels.astype(np.float))
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('PCA')
    plt.show()


def plotPCA2(X, labels, n_components=2):
    pca = PCA(n_components=n_components)
    XX = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print ('zachovana variancia pre PCA: ',var.sum())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for label in np.unique(labels):
        idx = np.where(labels == label)
        plt.scatter(XX[idx, 0], XX[idx, 1], c=colors[label])

    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('PCA')
    plt.show()


df = pd.read_csv('inputs/Airplane_Crashes_Since_1908.csv')
print(df['Summary'][0], '\n')
print(df['Summary'][1])

data = df['Summary'][:].fillna('unknown')


# vectorizer  = CountVectorizer(stop_words='english', max_features=2000)
# tr_sparse = vectorizer.fit_transform(data)
# # print(tr_sparse)
# X = tr_sparse.toarray()
# print(X)
# X = scale(X)
# print(len(X[0]))
# print(vectorizer.get_feature_names())

# #counts
# print(list(zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())))



# # n-gram
# bigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)
# tr_sparse = bigram_vectorizer.fit_transform(data)
# X = tr_sparse.toarray()
# # print(X)
# # print(len(X[0]))
# # print(bigram_vectorizer.get_feature_names())
# #counts
# print(list(zip(bigram_vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())))


# # lustering
# estimator = KMeans(n_clusters=7, random_state=8)
# estimator.fit(X)
# labels = estimator.labels_
# df['clus'] = labels
# print(df)


# plotPCA(X, labels, n_components=2)


# print(df['clus'].value_counts())


# clus_idx = df[df['clus'] == 5].index
# freq = X[clus_idx, :].sum(axis=0)
# df_words = pd.DataFrame({'word': bigram_vectorizer.get_feature_names(), 'freq': freq})
# print(df_words.sort_values('freq', ascending=False))


def get_clusters(df, text_feature, vectorizer, n_clusters, random_sate=10, plot_PCA=True):
    df[text_feature] = df[text_feature].fillna('unknown')

    tr_sparse = vectorizer.fit_transform(df[text_feature])
    X = tr_sparse.toarray()
    X = scale(X)
    # clustering
    estimator = KMeans(n_clusters=n_clusters, random_state=random_sate)
    estimator.fit(X)
    labels = estimator.labels_
    df['cluster'] = labels

    if plot_PCA:
        plotPCA2(X, labels, n_components=2)

    cluster_words = []
    for i in range(n_clusters):
        clus_idx = df[df['cluster'] == i].index
        freq = X[clus_idx, :].sum(axis=0)
        df_words = pd.DataFrame({'word': bigram_vectorizer.get_feature_names(), 'freq': freq})
        cluster_words.append(df_words.sort_values('freq', ascending=False).reset_index(drop=True))

    return df, cluster_words


# stop_words = text.ENGLISH_STOP_WORDS.union(['aircraft', 'plane', 'crash'])
bigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)
df, cluster_words = get_clusters(df, 'Summary', bigram_vectorizer, 7, random_sate=10)

# for clus in cluster_words:
#   print(clus.head(25))

# print('Pocetnost clustrov')
# print(df['cluster'].value_counts())



# import matplotlib.pyplot as plt
# from numpy.random import random

# colors = ['b', 'c', 'y', 'm', 'r']

# lo = plt.scatter(random(10), random(10), marker='x', color=colors[0])
# ll = plt.scatter(random(10), random(10), marker='o', color=colors[0])
# l  = plt.scatter(random(10), random(10), marker='o', color=colors[1])
# a  = plt.scatter(random(10), random(10), marker='o', color=colors[2])
# h  = plt.scatter(random(10), random(10), marker='o', color=colors[3])
# hh = plt.scatter(random(10), random(10), marker='o', color=colors[4])
# ho = plt.scatter(random(10), random(10), marker='x', color=colors[4])

# plt.legend((lo, ll, l, a, h, hh, ho),
#            ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
#            scatterpoints=1,
#            loc='lower left',
#            ncol=3,
#            fontsize=8)

# plt.show()