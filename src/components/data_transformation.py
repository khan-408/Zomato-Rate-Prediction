import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import FunctionTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('Getting data transsformation obj Initiated.')
            
            num_columns = ['votes', 'cost']
            cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'type']            
            all_columns = num_columns+cat_columns

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median'))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('label_encoder', OrdinalEncoder())
                ]
            )


            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipelihne',cat_pipeline,cat_columns)],
                remainder='passthrough')

            return preprocessor

            logging.info('Data Pipeline has been completed.')

        except Exception as e:
            logging.info('Error occured in running get_data_transformation_obj')
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test splitted data from artifactss
            train_df = pd.read_csv('artifacts/train.csv')
            test_df =  pd.read_csv('artifacts/test.csv')
            
            logging.info('reading train and test data is completed.')
            logging.info(f'Train DataFrame Head: \n {train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n {test_df.head().to_string()}')


            logging.info('Concatenating Train and test csvs')
            
            # Concatinating Tarin and Test data 
            df = pd.concat([train_df,test_df],axis=0)

            
            
            logging.info('Obtaining preprocessing object')

            
            #Rate handling
            df['rate'] = df['rate'].apply(lambda x: float(x.split('/')[0]) if (len(x)>3) else np.nan)

            #Cost handling
            df['cost'] =df['cost'].str.replace(',','').astype(float)

            logging.info(f'DataFrame Head: \n {df.head().to_string()}')
            df = df.dropna()

            # Defining Independent and dependent variables
            features = df.drop(['rate'],axis=1)
            target = df['rate']

            #Applying these pipelines on dataset specifically
            preprocessor_obj = self.get_data_transformation_obj()

            features = preprocessor_obj.fit_transform(features)
            # test_df1 = preprocessor_obj.fit_transform(test_df)
 
            num_columns = ['votes', 'cost']
            cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'type']            
            all_columns = num_columns+cat_columns
            # Independent features as Dataframe 
            features = pd.DataFrame(features, columns = all_columns)

            
            logging.info(f'DataFrame Head: \n {features.head().to_string()}')
            # logging.info(f'Test DataFrame Head: \n {df.head().to_string()}')

            
            
            logging.info("Applying preprocessing object on training and testing datasets.")

            
 
            #Saving the preprocessor.pkl object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            # logging.info("preprocessor pickle file saved.")
            logging.info('All sort of transformation has been done.')
            return (
                features,
                target,
                self.data_transformation_config.preprocessor_obj_file_path   
            )





        except Exception as e:
            logging.info('Error Occured in Initiating Data Transformation')
            raise CustomException(e,sys)