import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = ['id'])
    return df

def clean_data(df):
    # split the values in the categories column on the ';'
    #categories = df['categories'].str.split(';', expand = True)
    # use the first row of categories dataframe to create column names for the categories data
    #row = categories.iloc[0]
    #category_colnames = row.transform(lambda x: x[:-2]).tolist()
    #categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1.
    #for column in categories:
        # set each value to be the last character of the string
        #categories[column] = categories[column].transform(lambda x: x[-1:])
    
        # convert column from string to numeric
        #categories[column] = pd.to_numeric(categories[column])
    
    # replace categories column in df with the new category columns
    #df.drop('categories', axis = 1, inplace = True)
    
    
    # split the values in the categories column on the ';'
    categories = df.categories.str.split(';', expand=True)

    # use the first row of categories dataframe to create column names for the categories data
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: str(x)[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = pd.Series([str(x)[-1] for x in categories[column]])
        categories[column] = categories[column].astype(int)
    
    # replace categories column in df with the new category columns
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # Remove child_alone as it has all zeros
    df = df.drop(['child_alone'],axis=1)
    # There is a category 2 in 'related' column. This could be an error. 
    # In the absense of any information, we assume it to be 1 as the majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    return df



def save_data(df, database_filename):
    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    conn = sqlite3.connect('data/DisasterResponse.db')
    #df.to_sql('disaster_response_clean', con = conn, if_exists='replace', index=False)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, con = conn, if_exists='replace', index=False)


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