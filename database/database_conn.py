import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

def get_connection_uri():
    
    dbhost = os.environ.get('DB_HOST')
    dbname = os.environ.get('DB_NAME')
    dbuser = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD') 
    
    db_uri = f"host={dbhost} dbname={dbname} user={dbuser} password={password} sslmode=require"
    return db_uri # returns the connection string if the user has the env file


def connect_to_azure():
    conn_string = get_connection_uri() # get connection string first
    try:
        conn = psycopg.connect(conn_string) # try connecting to azure
        print("Connected to azure..")
        conn.close()
    except Exception as e:
        print(f"failed to connect: {e}")
        
if __name__ == "__main__":
    connect_to_azure()