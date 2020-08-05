# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 15:49:12 2020

@author: shobham
"""

import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
import pickle


nlp = spacy.load("en_core_web_sm")
   
class train_classification_model:
    
    def load_data(self):
        """Reading input data"""
        train_df = pd.read_csv("data/train.csv", header=None)
        train_df[1] = train_df[1] + " "+ train_df[2]
        train_df.drop(train_df.columns[2], axis=1, inplace=True)
        
        test_df = pd.read_csv("data/test.csv", header=None)
        test_df[1] = test_df[1] + " "+ test_df[2]
        test_df.drop(test_df.columns[2], axis=1, inplace=True)
        
        classes_df = pd.read_csv("data/classes.csv")
        return train_df, test_df, classes_df
    
    def clean_data(self, data):
        """Performing text cleaning operations on input data"""
        data=data.lower()
        doc=nlp(data, disable=['parser', 'ner'])
        
        #Removing stopwords, digits and punctuation from data
        tokens = [token.lemma_ for token in doc if not (token.is_stop
                                                 or token.is_digit
                                                  or token.is_punct
                                                 )]
   
        tokens = " ".join(tokens)
        return tokens
    
    
    
    def generate_avg_vector(self, data):
        """Generate Mean Vector using spacy word vectors"""
        doc=nlp(data)
        data_vector = [token.vector for token in doc]
        mean_vector = np.mean(data_vector, axis=0)
        return mean_vector
    
    
    def get_split_data(self, train_df, test_df):
        X_train = train_df[1]
        y_train = train_df[0]
        X_test = test_df[1]
        y_test = test_df[0]
        return X_train,X_test,y_train,y_test
        
    def train_svm_model(self, X_train, X_test, y_train, y_test):
        """Train model using CountVectorizer, TfidfTransformer, SVMClassifier"""
        clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                          ('clf', LinearSVC())])
        clf = clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print('Confusion matrix\n',confusion_matrix(y_test,pred))
        print('Classification_report\n',classification_report(y_test,pred))
        return clf
    
    
if __name__=="__main__":
    train_classification_model=train_classification_model()
    train_df, test_df, classes_df = train_classification_model.load_data()
    train_df[1] = train_df[1].apply(lambda x : train_classification_model.clean_data(x))
    test_df[1] = test_df[1].apply(lambda x : train_classification_model.clean_data(x))
    X_train,X_test,y_train,y_test=train_classification_model.get_split_data(train_df, test_df)
    clf=train_classification_model.train_svm_model(X_train, X_test, y_train, y_test)
    model_file = 'data/svm_model.pickle'
    pickle.dump(clf, open(model_file, 'wb'))
    
