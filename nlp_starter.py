import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib
matplotlib.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('inputs/Airplane_Crashes_Since_1908.csv')
print(df['Summary'][0], '\n')
print(df['Summary'][1])

tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(df['Summary'][:2])
print(tr_sparse)