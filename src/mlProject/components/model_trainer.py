import pandas as pd
import os
from mlProject import logger
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mlProject.entity.config_entity import ModelTrainerConfig




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        train_y = train_y.values
        test_y = test_data[[self.config.target_column]]
        y_reshaped = train_y.reshape(-1)


        lr = GradientBoostingRegressor(learning_rate=self.config.learning_rate, max_depth=self.config.max_depth,n_estimators=self.config.n_estimators,subsample=self.config.subsample, random_state=42)
        lr.fit(train_x, y_reshaped)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

