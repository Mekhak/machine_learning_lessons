
class UserParams():

	def __init__(self, dataset_path, target_name, train_test_path, test_size = 0.2, problem_type='auto'):
		self.dataset_path = dataset_path
		self.target_name = target_name
		self.train_test_path = train_test_path
		self.test_size = test_size
		self.problem_type = problem_type