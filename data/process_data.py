import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on="id", how="outer")

    return df

def clean_data(df):

    categories_temp = df['categories'].str.split(';', expand = True)

    colnames = categories_temp.iloc[0,:].apply(lambda x: x[:-2]).tolist()

    categories_temp.columns = colnames

    for col in colnames:
        categories_temp[col] = categories_temp[col].str[-1]
        categories_temp[col] = pd.to_numeric(categories_temp[col])

    df.drop('categories', inplace = True, axis = 1)

    df = pd.concat([df, categories_temp], axis = 1)

    df = df[~df.duplicated()]

    df['related'][df['related'] == 2] = df['related'].mode().iloc[0]

    return df


def save_data(df, database_filename):

    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('messages_and_categories', engine, index = False,
    if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
