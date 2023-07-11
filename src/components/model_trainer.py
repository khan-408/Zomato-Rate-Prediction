import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
from src.utils import save_object
from dataclasses import dataclass
# Importing Models 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  ExtraTreesRegressor


@dataclass
class ModelTrainerConfig():
    model_trainer_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,features,target):
        try:
            logging.info('Defining Dependent and Independent features')
            X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=42)
            
            models = {
                'Linear_Reg' : LinearRegression(),
                'DTree' : DecisionTreeRegressor(min_samples_leaf=.0001),
                'RForest' : RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001),
                'RForest1' : RandomForestRegressor(n_estimators=100,random_state=329,min_samples_leaf=.0001),
                'ETree' : ExtraTreesRegressor(n_estimators = 100)

            }

            model_report:dict = evaluate_model(X_train, X_test, y_train, y_test,models)
            print(model_report)
            print('='*100)
            print(f'Model Report {model_report}.')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.model_trainer_path,
                 obj=best_model
            )



        except Exception as e:
            logging.info('Error occured in Initiating model Trainer')
            raise CustomException(e,sys)