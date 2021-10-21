import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    engine = create_engine("sqlite:///{}".format(database_filepath))

    df = pd.read_sql_table('messages_and_categories', con = engine)

    X = df['message']
    Y = df.iloc[:,4:]

    category_names = Y.columns

    return X, Y, category_names




def tokenize(text):
    # Normalize text to standard characters

    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    lemmatised_tokens = []
    for tok in tokens:
        lemmatised_tok = lemmatizer.lemmatize(tok)
        lemmatised_tokens.append(lemmatised_tok)

    return lemmatised_tokens




def build_model():


    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
          #'clf__estimator__class_weight': ({0:1, 1:1}, {0:1, 1:2}, {0:1, 1:5}),
        'clf__estimator__min_samples_leaf': [1, 2, 3],
        'clf__estimator__min_samples_split': [2, 4]}

    model = GridSearchCV(pipeline, param_grid = parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)
    print(model.best_params_)
    counter = 0
    for category in category_names:
        print("Statistics for {}".format(category))
        print(classification_report(Y_pred[:,counter], Y_test[category]))
        counter += 1


def save_model(model, model_filepath):
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
