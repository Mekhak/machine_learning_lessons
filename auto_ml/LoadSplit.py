import pandas as pd
from sklearn.model_selection import train_test_split
from enum import Enum
from Confirmator import Confirmator

class LoadSplit:

	def __init__(self, user_params):
		self.user_params = user_params

	ProblemType = Enum("ProblemType", "Regression Classification")

	def load_and_split(self):
		db = pd.read_csv(self.user_params.dataset_path)

		db.dropna(subset=[self.user_params.target_name])

		print(db.info(verbose = True))
		print()
		probem_type = self.check_problem_type(db)
		print("Problem type: ", probem_type.name, "\n")

		Confirmator.confirm()

		y = db.pop(self.user_params.target_name)
		X_train, X_test, y_train, y_test = train_test_split(db, y, test_size = self.user_params.test_size, random_state = 42)

		# save the train and split datasets
		X_train.to_csv(self.user_params.train_test_path + 'X_train.csv')
		X_test.to_csv(self.user_params.train_test_path + 'X_test.csv')
		y_train.to_csv(self.user_params.train_test_path + 'y_train.csv')
		y_test.to_csv(self.user_params.train_test_path + 'y_test.csv')

		return X_train, X_test, y_train, y_test


	def check_problem_type(self, data):
		if self.user_params.problem_type == 'regression':
			return LoadSplit.ProblemType.Regression

		if self.user_params.problem_type == 'classification':
			return LoadSplit.ProblemType.Classification

		if self.user_params.problem_type == 'auto':
			if data[self.user_params.target_name].dtype.kind == 'O':
				return LoadSplit.ProblemType.Classification

			print("target columns unique value count: ", len(data[self.user_params.target_name].unique()))

			if len(data[self.user_params.target_name].unique()) <= 10:
				return LoadSplit.ProblemType.Classification

			return LoadSplit.ProblemType.Regression
