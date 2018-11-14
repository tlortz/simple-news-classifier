"""
A basic news classification module that, once instantiated and fit, can be 
called repeatedly to classify incoming news articles. It hinges on the 
'20 newsgroups' dataset from scikit-learn

Sample usage: 

from news_classifier import *
classifier = News_Classifier()
classifier.setup()
classifier.fit()
classifier.predict([<news sample 1>, <news sample 2>, ...])

The setup and fit methods can be made to run more quickly by selecting a subset
of the categories in the 20 newsgroups dataset, e.g. 
['alt.atheism', 'talk.politics.misc', 'talk.religion.misc']

Also, the fit method can be run in different ways using different grid search params
There are some good examples at 
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

One possible set of options is built in as the default; here is another:
{'vect__max_df': (0.5, 0.75, 1.0),'vect__max_features': (None, 5000, 10000, 50000),\
'vect__ngram_range': ((1, 1), (1, 2)),'tfidf__use_idf': (True, False),\
'tfidf__norm': ('l1', 'l2'),'clf__alpha': (0.00001, 0.000001),\
'clf__penalty': ('l2', 'elasticnet'),'clf__n_iter': (10, 50, 80)}
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class News_Classifier():
    
    def __init__(self):
        self.model = None
        self.data = None
        self.labels = None
        self.categories = None
    
    def setup(self,categories=None):
        self.categories = categories
        self.data = fetch_20newsgroups(subset='train', categories=self.categories)
        self.labels = {}
        for n in range(len(self.data.target_names)):
            self.labels[n] = self.data.target_names[n]
        
    def fit(self,search_params={'vect__max_df': (0.5, 0.75, 1.0),\
                                                'vect__ngram_range': ((1, 1), (1, 2)),\
                                                'tfidf__use_idf': (True, False),\
                                                'clf__alpha': (0.00001, 0.000001),\
                                                'clf__penalty': ('l2', 'elasticnet')}):
        pipeline = Pipeline([('vect', CountVectorizer()),\
            ('tfidf', TfidfTransformer()),\
            ('clf', SGDClassifier()),\
        ])
        grid_search = GridSearchCV(pipeline, search_params, n_jobs=-1, verbose=0)
        grid_search.fit(self.data.data, self.data.target)
        self.model = grid_search
        
        
    def predict(self,news_articles):
        # news_articles should be a list of strings
        predictions = self.model.predict(news_articles)
        results = []
        for p in predictions:
            results.append(self.labels[p])
        return results
