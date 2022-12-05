import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    Combines two csv files with cleaning
    Parameters:
        messages_filepath: Path of the first file
        categories_filepath: Path of the second file
        
    Returns:
        df: cleaned df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    
    messages = messages.drop_duplicates(subset="id")
    categories = categories.drop_duplicates(subset="id")
    df = pd.merge(messages, categories, how="outer", on="id")
    
    temp_df = categories["categories"].str.split(";", expand=True)
    values = categories.iloc[0].str.split(";")[1]
    name_of_columns = []
    for v in values:
        name_of_columns.append(re.findall('([^-]+)', v)[0])

    temp_df.columns = name_of_columns

    for c in temp_df.columns:
        temp_df[c] = temp_df[c].apply(lambda x: re.findall('([^-]+)', x)[1])
        
    categories = temp_df
    
    for column in categories:

        categories[column] = categories[column].astype(int)
    df.drop(columns="categories", inplace=True)
 
    df.reset_index(inplace=True)
    
    categories.reset_index(inplace=True)
    df.drop(columns="index")
    df = pd.concat([df, categories], axis=1)
    df = df[df["related"] != 2]
    
    return df
def clean_data(df):
    '''
    Drop duplicates of the given df and return it
    '''
    df.drop_duplicates(inplace=True)
    df.drop(columns="child_alone", inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Saves formatted data to SQL database
    
    Parameters:
        df: the dataframe which keeps the table content
        database_filename: Name of the sqlite database file
    '''

    
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('disasters', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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