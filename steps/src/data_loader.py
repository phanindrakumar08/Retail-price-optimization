import pandas as pd
from sqlalchemy import create_engine, exc
from data.management.index import engine, create_engine, Session, connection

class DataLoader:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None

    def load_data(self, table_name: str)-> pd.DataFrame:
        query = "SELECT * FROM " + table_name
        try:
            self.data = pd.read_sql(query, self.engine)
            return self.data
        except exc.SQLAlchemyError as e:
            raise e
    
    def get_data(self)-> pd.DataFrame:
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data logged yet. Please run load_data first")
