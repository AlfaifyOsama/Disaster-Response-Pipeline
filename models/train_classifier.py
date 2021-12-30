# import libraries
import sys
import sqlite3
from sqlalchemy import create_engine
import re
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    This function loads the content of messages and categories 
    from SQLite database into variables X and y.
    Input:
    - database_filepath(String): location of the database file
    Output:
    - X(Dataframe): input messages
    - y(Dataframe): message labels
    - category_names(list): list of category names for messages
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("DisasterResponse", engine)
    X = df.iloc[:,1]
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    This function tokenizes the text, normalize, lemmatizes it, and change text to lower case.
    Input:
    - text(String): input text
    Output:
    - clean_tokens(String): tokenized and lemmatized text
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #stop words
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize(text)
    
    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
   
    return words_lemmed


def build_model(X,y):
    """
    This function defines a ML pipeline with parameters found using GridSearch.
    (The GridSearch part is done in the attached Jupyter notebook, since GridSearch
    takes a long time to run)
    Input:
    - X(Dataframe): input messages
    - y(Dataframe): message labels
    - 
    Output:
    - trained gridSearch model: ML model
    """
    #creating the pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    #create parameters for gridsearch
    parameters = {"clf__estimator__n_estimators":[100],
              "clf__estimator__min_samples_split":[2]}
    
    #Initialize the gridseachcv object 
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=2)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    This function evaluates the ML models using F1 score.
    Input:
    - model: a ML model
    - X_test(Dataframe): the test data
    - y_test(Dataframe): the test labels
    - category_names(list): list of category names for messages
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)

def save_model(model, model_filepath):
    """
    This function saves the trained model inpto a pickle file
    for production.
    Input:
    - model: a ML model
    - model_filepath: location to save the model
    """
    with open(model_filepath, "wb") as my_stream:
        pickle.dump(model, my_stream)

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X,y)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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