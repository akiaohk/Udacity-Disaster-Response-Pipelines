"""
Project: Disaster Response Pipeline

Script Syntax for execution:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
"""

# Import libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import time
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Loads data from SQL Database
    
    Arguments:
    database_filepath: SQL database file
    
    Outputs:
    X pandas_dataframe: Features dataframe
    Y pandas_dataframe: Target dataframe
    category_names list: Target labels 
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = database_filepath.replace(".db","") + "_table"
    df = pd.read_sql_table(table_name, con = engine)
    X,Y = df['message'], df.iloc[:,4:]

    # Y['related'] contains three distinct values
    # mapping extra values to `1`
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns

    return X, Y, category_names 

def tokenize(text):
    """
    Tokenizes text data
    
    Arguments:
    text str: Messages as text data
    
    Outputs:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """
    Build model with GridSearchCV
    
    Outputs:
    Trained model after performing grid search
    """
    
    # model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model function.
    
    Arguments:
        model -> sklearn model
        X_test -> Features for the test set
        Y_test -> Labels for the test set
        category_names -> list of category names
    """

    Y_prediction_test = model.predict(X_test)

    # Classification report
    print("Classification report")
    print(classification_report(Y_test, Y_prediction_test, target_names=category_names))

def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file 
    
    Arguments:
    model: Trained model
    model_filepath: Filepath to save the model
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle    
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
