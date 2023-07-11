import os
import sys
from src.utils import save_object,load_object
from src.logger import logging
from src.exception import CustomException
import pandas as pd
num_columns = ['votes', 'cost']
cat_columns = ['online_order', 'book_table', 'location', 'rest_type', 'cuisines', 'type']            
all_columns = num_columns+cat_columns
class PredictionPipeline:
            
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled_data = preprocessor.transform(features)
            # print(scaled_data)
            df = pd.DataFrame(scaled_data,columns=all_columns)
            # print(df)
            pred = model.predict(df)
            # print(pred)
             
            return pred



        except Exception as e:
            logging.info('Error occured in Prediction')
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                votes:float,
                cost:float,
                online_order:bool,
                book_table:bool,
                location:str,
                rest_type:str,
                cuisines:str,
                type:str):

        self.votes = votes
        self.cost = cost
        self.online_order = online_order
        self.book_table = book_table
        self.location = location
        self.rest_type = rest_type
        self.cuisines = cuisines
        self.type = type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                'votes':[self.votes],
                'cost':[self.cost],
                'online_order':[self.online_order],
                'book_table':[self.book_table],
                'location':[self.location],
                'rest_type':[self.rest_type],
                'cuisines':[self.cuisines],
                'type':[self.type]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            # print(df)
            return df

        
        except Exception as e:
            logging.info('Error occured in get data as dataframe')
            raise CustomException(e,sys)



if __name__=='__main__':
    predict_obj = PredictionPipeline()
    data  = CustomData(300,1000,False, True,"St. Marks Road", 
                "Fine Dining, Bar","South Indian, North Indian, Chinese, Street Food","Dine-out")
    df = data.get_data_as_dataframe()
    print(df)
    print(predict_obj.predict(df))
