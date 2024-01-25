import pandas as pd
import os
import sqlite3 as db
from sqlite3 import Error
from dotenv import load_dotenv

def main():

    load_dotenv()

    DATA_PATH = os.getenv("DATA_PATH")
    DATABASE_NAME = os.getenv("DATABASE_NAME")
    database_path = os.path.join(DATA_PATH, DATABASE_NAME)

    ### Create the Database ###
    conn = None
    try:
        conn = db.connect(database_path)
        print(db.version)
    except Error as e:
        print(e)
        print("Failed to create Database")
    finally:
        if conn:
            conn.close()
    print("Created database")

    ### Create the usr_prompts table ###
    conn = None
    try:
        conn = db.connect(database_path)
        usr_prompts_df = pd.read_csv(os.path.join(DATA_PATH, 'usr_prompts.csv'))
        # columns have to be: (uid, prompt, exptUserId) otherwise it will break
        usr_prompts_df.to_sql('usr_prompts', conn, if_exists='append', index=False)
    except Error as e:
        print(e)
        print("Failed to create the usr_prompts table")
    finally:
        if conn:
            conn.close()
    print("Created usr_prompts table")

    ### Create the usr_interactions table ###
    conn = None
    try:
        conn = db.connect(database_path)
        create_table_sql =  """CREATE TABLE IF NOT EXISTS usr_interactions (
                                exptUserId integer NOT NULL,
                                tstamp text NOT NULL,
                                history text,
                                PRIMARY KEY (exptUserId, tstamp),
                                FOREIGN KEY (exptUserId) REFERENCES usr_prompts (exptUserId)
                            );"""
        cur = conn.cursor()
        cur.execute(create_table_sql)
    except Error as e:
        print(e)
        print("Failed to create usr_intractions table")
    finally:
        if conn:
            conn.close()
    print("Created usr_interactions table")
    print("Finished Database Set-up")

if __name__ == "__main__":
    main()