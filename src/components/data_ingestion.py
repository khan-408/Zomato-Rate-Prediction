from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.utils import evaluate_model

import sys
import os
import pandas as pd

#Defining columns name in order to apply operations 
drop_column = ['url','name','phone','dish_liked','address','reviews_list','menu_item','listed_in(city)']
rename_column = {'listed_in(type)':'type','approx_cost(for two people)':'cost'}

# Creating a DataClass
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        # To initialize the DataClass (DataIngestionConfig)
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has been started.')


        try:
            #Reading Csv as Dataframe
            df = pd.read_csv(os.path.join('notebooks/data','zomato.csv'))
            logging.info('Data has been read.')
            # To create a directory to save dataset as Raw Csv
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            # Gen_pipeline is a pipeline to applying some sort of basic transformation on df
            gen_pipeline = Pipeline(
                steps=[
                    ('drop_columns',FunctionTransformer(lambda x:x.drop(drop_column,axis=1),validate=False)),
                    ('drop_na',FunctionTransformer(lambda x:x.dropna(how='any'))),
                    ('dropping_duplicates',FunctionTransformer(lambda x: x.drop_duplicates())),
                    ('renaming_col',FunctionTransformer(lambda x: x.rename(columns=rename_column))),
                    ('replace_into_bool',FunctionTransformer(lambda x: x.replace(('Yes','No'),(True,False))))
                                
                ]
            )
            df = gen_pipeline.fit_transform(df)
            
            logging.info('Dropping columns,Dropping Missing values and duplicates has been done')
            

            logging.info('Train_Test_Split')
            # In order to save train and test splitted data into artifacts.
            train_set,test_set = train_test_split(df,test_size=.25,random_state=42)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False)

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info('Ingestion of Data has been completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Error occured while initiating data ingestion')
            raise CustomException(e,sys)

    '''

if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)
    '''