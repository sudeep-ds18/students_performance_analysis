import sys
import pandas as pd
import numpy as np 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass 
class DataTransformationConfig():
    preprocessor_obj_file_path= os.path.join("artifacts","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
           num_atrribs = ["writing_score","reading_score"]
           cat_atrribs = ["gender","race_ethnicity","parental_level_of_education",
                          "lunch","test_preparation_course"]
           
           num_pipeline=Pipeline(
               steps=[
                   ("imputer",SimpleImputer(strategy="median")),
                  ("scaler",StandardScaler())
                ])

           cat_pipeline=Pipeline(
               steps=[
               ("imputer",SimpleImputer(strategy="most_frequent")),
               ("onehot",OneHotEncoder(handle_unknown="ignore"))
               
                 ])

           logging.info("Build pipeline for num,cat")

           preprocessor=ColumnTransformer(
               
               transformers=[
               ("num_pipeline",num_pipeline,num_atrribs),
               ("cat_pipeline",cat_pipeline,cat_atrribs)
                     ]
           )


           return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
            try:
                '''get train and test data 
                separate target and input feature 
                
                '''
                train_df= pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                logging.info("read training and testing data successfully")

                preprocessing_obj = self.get_data_transformer_obj()

                target_column = "math_score"

                input_feature_train_df=train_df.drop(columns=[target_column])
                target_feature_train_df=train_df[target_column]

                input_feature_test_df=test_df.drop(columns=[target_column])
                target_feature_test_df=test_df[target_column]

                logging.info("Applying preprocessing obj on training and testing dataframe")

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                         input_feature_train_arr,
                         target_feature_train_df.to_numpy().reshape(-1, 1)
                        ]

                test_arr = np.c_[
                        input_feature_test_arr,
                        target_feature_test_df.to_numpy().reshape(-1, 1)
                        ]
                
                save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
                logging.info("preprocessor saved successfully")
                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:    
                raise CustomException(e,sys) 

if __name__ == "__main__":
    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )
    )

    print("Data Transformation completed")
    print("Preprocessor saved at:", preprocessor_path)
        

        




