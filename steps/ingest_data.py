import logging 
import pandas as pd
from zenml import step
from steps.src.data_loader import DataLoader

@step(enable_cache = False)
def ingest(
    table_name: str, 
    for_predict : bool=False
) -> pd.DataFrame:
    try:
        data_loader = DataLoader("postgresql://postgres:12345678@localhost:5432/retail")
        data_loader.load_data(table_name) 
        if for_predict:
            df.drop(columns = ['unit_price'], inplace = True)
        df = data_loader.get_data()
        logging.info(f"Successfully read data from {table_name}")
        return df

    except Exception as e:
        logging.error(f"Error while reading data from {table_name}")
        raise e
    
