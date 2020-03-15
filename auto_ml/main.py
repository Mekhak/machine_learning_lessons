from LoadSplit import LoadSplit
from DataPreprocessor import DataPreprocessor
from UserParams import UserParams

# load and split the data
user_params = UserParams(dataset_path="data\\santander_customer_transaction_prediction_target.csv",
                         target_name="target",
                         train_test_path="train_test_splited\\")

load_splitter = LoadSplit(user_params)
X_train, X_test, Y_train, Y_test = load_splitter.load_and_split()

# preprocess the data
data_preprocessor = DataPreprocessor(user_params, X_train, X_test, Y_train, Y_test)
X_train, X_test = data_preprocessor.fit_transform()