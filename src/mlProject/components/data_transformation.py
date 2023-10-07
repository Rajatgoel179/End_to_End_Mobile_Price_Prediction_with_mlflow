import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlProject.entity.config_entity import DataTransformationConfig






class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def clean_data(self):
            df = pd.read_csv(self.config.data_path)
            
            np.array(df["Screen Size (inches)"])
            df["Price ($)"]=df["Price ($)"].str.extract('(\d+)', expand=False)
            df["Price ($)"]=df["Price ($)"].astype(int)
            df["RAM "]=df["RAM "].str.replace("GB","")
            df["RAM "]=df["RAM "].astype(int)


            df["Storage "]=df["Storage "].str.replace("GB","")
            df["Storage "]=df["Storage "].astype(int)
            df['n_cameras'] = df['Camera (MP)'].str.count('\\+') + 1
            res1 = []
            res2 = []
            res3 = []
            res4 = []
            for x in df['Camera (MP)']:
                 resolutions = x.split('+')
                 tam = len(resolutions)
                 if tam == 1:
                      res1.append(resolutions[0])
                      res2.append('0')
                      res3.append('0')
                      res4.append('0')
    
                 if tam == 2:
                      res1.append(resolutions[0])
                      res2.append(resolutions[1])
                      res3.append('0')
                      res4.append('0')
    
                 if tam == 3:
                      res1.append(resolutions[0])
                      res2.append(resolutions[1])
                      res3.append(resolutions[2])
                      res4.append('0')
    
                 if tam == 4:
                      res1.append(resolutions[0])
                      res2.append(resolutions[1])
                      res3.append(resolutions[2])
                      res4.append(resolutions[3])
    
            df['res1'] = res1
            df['res2'] = res2
            df['res3'] = res3
            df['res4'] = res4

            df= df.drop(columns='Camera (MP)')

            df['Screen Size (inches)'].replace(regex=True, inplace=True, to_replace=r'[^0-9.\-]', value=r'')
            cem1 = []
            cem2 = []
            cem3 = []
            for x in df['Screen Size (inches)']:
                 resolutions = x.split('.')
                 tam = len(resolutions)
                 if tam == 1:
                      cem1.append(resolutions[0])
                      cem2.append('0')
                      cem3.append('0')
    
                 if tam == 2:
                      cem1.append(resolutions[0])
                      cem2.append(resolutions[1])
                      cem3.append('0')
                 if tam == 3:
                      cem1.append(resolutions[0])
                      cem2.append(resolutions[1])
                      cem3.append(resolutions[2])
    
    
            df['cem1'] = cem1
            df['cem2'] = cem2
            df['cem3'] = cem3




            df= df.drop(columns='Screen Size (inches)')
            df["screen"] = df['cem1']+"."+ df["cem2"]
            df=df.drop(["cem1","cem2","cem3"],axis=1)

            df["screen"]=df["screen"].astype(float)
            df["res1"]=df["res1"].str.extract('(\d+)', expand=False)
            df["res1"]=df["res1"].astype(int)
            df["res2"]=df["res2"].str.extract('(\d+)', expand=False)
            df["res2"]=df["res2"].astype(int)
            df["res3"]=df["res3"].str.extract('(\d+)', expand=False)
            df["res3"]=df["res3"].astype(int)
            np.array(df["res4"])
            df["res4"]=df["res4"].str.extract('(\d+)', expand=False)
            df['res4'].isnull().sum()
            df['res4'] = df['res4'].fillna(0)
            df['res4'] = df['res4'].astype(int)


            label_encoder = LabelEncoder()
            df['Brand'] = label_encoder.fit_transform(df['Brand'])
            df = df.drop("Model", axis=1)
            return df
    

    def train_test_spliting(self,df):

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(df, test_size=0.25, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        