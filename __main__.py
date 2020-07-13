import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/yelp.csv')
df.head()
df.info()
df.describe()

df['length'] = df['text'].apply(len)
df.describe()

df[df['length'] == 4997]
df.loc[55]['text']

sns.set_style('darkgrid')
sns.FacetGrid(df, col='stars').map(plt.hist, 'length')

sns.boxplot(x='stars', y='length', data=df)

sns.countplot(x='stars', data=df)

by_stars = df.groupby('stars')
mean_by_stars = by_stars.mean()
corr_by_stars = mean_by_stars.corr()
sns.heatmap(corr_by_stars, cmap='plasma', annot=True)

yelp_class = df[(df['stars'] == 1) | (df['stars'] == 5)]
yelp_class.head()

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=101)

X_train.shape
X_test.shape

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

X = yelp_class['text']
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=101)

pipe.fit(X, y)

y_pred = pipe.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
