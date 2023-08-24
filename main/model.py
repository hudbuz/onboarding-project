import os
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, InferenceOutput, ModelSchema
from qwak.model.adapters import JsonInputAdapter, JsonOutputAdapter
from qwak.feature_store.offline.client import OfflineClient




class CancerDetectionModel(QwakModel):

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="radius_mean", type=float),
                ExplicitFeature(name="texture_mean", type=float),
                ExplicitFeature(name="perimeter_mean", type=float),
                ExplicitFeature(name="area_mean", type=float),
                ExplicitFeature(name="smoothness_mean", type=float),
                ExplicitFeature(name="compactness_mean", type=float),
                ExplicitFeature(name="concavity_mean", type=float),
                ExplicitFeature(name="symmetry_mean", type=float),
                ExplicitFeature(name="fractal_dimension_mean", type=float),
            ],
            outputs=[
                InferenceOutput(name="diagnosis", type=int)
            ])
  
    
    def build(self):
        # Read data
        # file_absolute_path = os.path.dirname(os.path.abspath(__file__))
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=2)
        df = self.get_feature_data(start_time, end_time)

        clean_df = self.clean_data(df)
        features_mean = [feature.name for feature in self.schema().inputs]
        print("build features",features_mean)
        outcome_var='diagnosis'
        model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
        model.fit(clean_df[features_mean],clean_df[outcome_var])
        predictions = model.predict(clean_df[features_mean])
        print('predictions', predictions)
        accuracy = metrics.accuracy_score(predictions,clean_df[outcome_var])

        #Perform k-fold cross-validation with 5 folds
        kf = KFold(n_splits=5)
        error = []
        for train, test in kf.split(clean_df):
            # Filter training data
            train_predictors = (clean_df[features_mean].iloc[train,:])
            
            # The target we're using to train the algorithm.
            train_target = clean_df[outcome_var].iloc[train]
            
            # Training the algorithm using the predictors and target.
            model.fit(train_predictors, train_target)

            error.append(model.score(clean_df[features_mean].iloc[test,:], clean_df[outcome_var].iloc[test]))
            
            print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
            
        #Fit the model again so that it can be refered outside the function:
        model.fit(clean_df[features_mean],clean_df[outcome_var])
        self.model = model
        return model

    def get_feature_data(self, start_time: datetime.datetime, end_time: datetime.datetime) -> pd.DataFrame:
        offline_client = OfflineClient()
        key_to_features = {
            'id': [
                "breast-cancer-ingestion.radius_mean",
                "breast-cancer-ingestion.texture_mean",
                "breast-cancer-ingestion.perimeter_mean",
                "breast-cancer-ingestion.area_mean",
                "breast-cancer-ingestion.smoothness_mean",
                "breast-cancer-ingestion.compactness_mean",
                "breast-cancer-ingestion.concavity_mean",
                "breast-cancer-ingestion.symmetry_mean",
                "breast-cancer-ingestion.fractal_dimension_mean",
                "breast-cancer-ingestion.diagnosis"
                ]
        }
        return offline_client.get_feature_range_values(
            entity_key_to_features=key_to_features,
            start_date=start_time,
            end_date=end_time,
        )


    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features_mean=list(df.columns)
        return pd.DataFrame(
                    self.model.predict(df[features_mean]),
            columns=['diagnosis']
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame: 
        # Remove data
        df.drop('id',axis=1,inplace=True)
        df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

        return df