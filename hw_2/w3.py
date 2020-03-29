class MyLinearRegressor:
    def __init__(self, d, target, alpha=0.003):
        self.data = d
        self.target = target
        self.alpha = alpha
        self.m = self.data.shape[0]
        self.ncols = self.data.shape[1]
        self.W = self.w_ = np.random.normal(loc= 0.0, scale = 0.01, size = self.ncols)
        print("initial weights: ", self.W)

    def predict(self, X):
        return X.dot(self.W)

    def mse(self):
        return sum((self.predict(self.data) - self.target) ** 2) / self.m

    def derivative(self):
        # derivative of mean squared error function
        errors = self.predict(self.data) - self.target
        return self.data.T.dot(errors)

    def fit(self, num_of_iterations=100):
        for i in range(num_of_iterations):
            self.W -= self.alpha * self.derivative()
        print("final weights:   ", self.W)





bias = np.ones(100)
x1 = np.random.randint(low=1, high=20, size=100)
x2 = np.random.randint(low=1, high=20, size=100)
x3 = np.random.randint(low=1, high=20, size=100)
x4 = np.random.randint(low=1, high=20, size=100)

w0 = 1
w1 = 7
w2 = 3
w3 = 9
w4 = 6

X = np.column_stack((bias, x1, x2, x3, x4))

# def standardize(X):
#     X_std = X.copy()
#     for i in range(1, X.shape[1]):
#         X_std[:, i] = (X_std[:, i] - X_std[:, i].mean()) / X_std[:, i].std()
#     return X_std

# X_std = standardize(X)
y = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4


lr = MyLinearRegressor(X, y, alpha = 0.00004)
lr.fit(num_of_iterations=4000)
print("actual weights:  ", [w0, w1, w2, w3, w4])