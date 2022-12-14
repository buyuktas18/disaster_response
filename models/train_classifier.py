import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle
import re

def load_data(database_filepath):
    
    '''
    Loads the data from the given filepath
    
    Parameters:
        database_filepath (str): The path of the database and file
        
    Returns:
        X (list): Featureset, list of our inputs
        Y (list): Target values
        column values (list): list of categories wanted to be classified
    '''
    
    engine = create_engine('sqlite:///' + database_filepath + '.db')
    df = pd.read_sql(f"SELECT * FROM disasters", engine)
    
    
    X = df["message"]
    Y = df.iloc[:,5:]
    
    print(Y.columns)
    
    for c in Y:
        if Y[c].values.tolist().count(1) == 0:
            Y.drop(columns=c, inplace=True)
            print(c)
    
   
    

    return X, Y, list(Y.columns.values)

def tokenize(text):
    
    '''
    Tokenizer of the NLP pipeline, returns the tokenized words
    
    Parameters:
        text (str): Containing a lot of words as one or more sentence
        
    Returns:
        clean_tokens (list): List of seperate words
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    Returns the pipeline with tuning
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    
    parameters = {
        
        'clf__estimator__penalty': ["l1","l2"],
        'clf__estimator__C' : np.logspace(-4, 4, 50)
        
    }
    

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Prints different evaluation metrics with respect to given model results
    
    Parameters:
        model: Machine Learning model
        X_test: test data for features
        Y_test: test data for targets
        category_names: list of categories
    '''
    
    y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):
        if Y_test[c].values.tolist().count(1) > 0 and list(y_pred[:,i]).count(1) > 0:
            
            acc = accuracy_score(y_pred[:,i], Y_test[c].values.tolist())
            f1 = f1_score(y_pred[:,i], Y_test[c].values.tolist())
            p = precision_score(y_pred[:,i], Y_test[c].values.tolist())
            r = recall_score(y_pred[:,i], Y_test[c].values.tolist())
        
            print(f"for column {c}: ")
            print('\t' + f"Accuracy: {acc}")
            print('\t' + f"F1 score: {f1}") 
            print('\t' + f"Precision: {p}") 
            print('\t' + f"Recall: {r}") 
        else:
            print("This column couln't be evaluated since it is not containing any true sample")
    
    

def save_model(model, model_filepath):
    '''
    Saves model permanently
    Parameters:
        model: The machine learning model
        model_filepath: The path of the model that will be saved
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
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