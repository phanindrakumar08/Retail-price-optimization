import pandas as pd
from zenml import step
from zenml.logger import get_logger
from steps.src.data_processor import CategoricalEncoder
from steps.src.feature_engineering import DateFeatureEngineer, FeatureEngineer

logger = get_logger(__name__)


@step(enable_cache = True)
def categorical_encode(df:pd.DataFrame)-> pd.DataFrame:
    try:
        encoder = CategoricalEncoder(method= "onehot")
        df = encoder.fit_transform(df, columns=['product_id', 'product_category_name'])
        logger.info("Successfully encoded categorical variables")
        return df
    except Exception as e:
        logger.error('Error while encoding, please give onehot or ordinal')
        raise e
    

@step(enable_cache = True)
def feature_engineer(df:pd.DataFrame )-> pd.DataFrame:
    try:
        date_engineer = DateFeatureEngineer(date_format= "%d-%m-%Y")
        df_transformed = date_engineer.fit_transform(df, columns = ["month_year"])
        logger.info("Successfully engineered features")
        
        df_transformed.drop(['id', 'month_year'], axis = 1, inplace = True)
        return df_transformed
    
    except Exception as e: 
        raise e