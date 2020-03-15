import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from Confirmator import Confirmator

class DataPreprocessor:

    def __init__(self, user_params, X_train, X_test, Y_train, Y_test):
        self.user_params = user_params
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    def fit_transform(self):
        # seperate numeric and categorical features
        numerics = self.X_train.select_dtypes(include=np.number).columns.tolist()
        categoricals = list(set(self.X_train.columns) - set(numerics))

        print("numericals:  ", numerics)
        print("categorcals: ", categoricals)

        Confirmator.confirm()

        # define pipelines for handling missing values
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median"))
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value='other')),
            ('encoder', OrdinalEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, numerics),
            ("cat", cat_pipeline, categoricals)
        ])

        self.handle_unknowns_in_test_set(categoricals)
        X_train_prepared = full_pipeline.fit_transform(self.X_train)
        X_test_prepared = full_pipeline.transform(self.X_test)

        return X_train_prepared, X_test_prepared

    # TODO: define transform method later to transform validation data

    def handle_unknowns_in_test_set(self, categoricals):
        for c in categoricals:
            cat_train = set(self.X_train[c].unique())
            cat_test = set(self.X_test[c].unique())
            unknowns = cat_test - cat_train
            # print("unknowns before: ", unknowns)
            if unknowns:
                # print("other in train: ", 'other' in cat_train)
                fill_val = 'other' if 'other' in cat_train else self.X_train[c].mode().values[0]
                self.X_test.replace(list(unknowns), fill_val, inplace=True)

            # cat_train = set(self.X_train[c].unique())
            # cat_test = set(self.X_test[c].unique())
            # unknowns = cat_test - cat_train
            # print("unknowns after: ", unknowns)