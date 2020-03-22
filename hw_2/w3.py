import pandas as pd
import numpy as np
import math



class MyLinearRegressor:
    def __init__(self, d, target, alpha=0.0003):
        self.data = d
        self.target = target
        self.alpha = alpha
        self.m = d.shape[0]
        self.ncols = d.shape[1]
        self.W = [1] * self.ncols

    def predict(self, x):
        res = 0
        for w_, x_ in zip(self.W, x):
            res += w_ * x_
        return res

    def rmse(self):
        rmse = 0
        for i in range(self.m):
            rmse += math.pow(self.predict(self.data.iloc[i].to_list()) - self.target[i], 2)
        rmse = math.sqrt(rmse / (2 * self.m))
        return rmse

    def derivative(self, i):
        res = 0
        for r in range(self.m):
            res += (self.predict(self.data.iloc[r].to_list()) - self.target[r]) * self.data.iloc[r, i]
        return res / self.m

    def fit(self, num_of_iterations=100):
        for i in range(num_of_iterations):
            # for updating weights simultaneously
            new_weights = []
            for j in range(self.ncols):
                new_weights.append(self.W[j] - self.alpha * self.derivative(j))
                # self.W[j] = self.W[j] - self.alpha * self.derivative(j)
            self.W = new_weights
        print(self.W)




dataset_path = "data\\sberbank_russian_housing_market_price_doc.csv"

data = pd.read_csv(dataset_path)
data.fillna(value=data.mean(), inplace=True)
target = data.price_doc
data = data[['floor', 'num_room', 'kitch_sq', 'material']]

opstimizer = MyLinearRegressor(data, target)
optimizer.fit()


# g - is the function that we want to learn from our sample that has a general structure and loss function
# ------------------------------------- specify the general structure
# the exact type of true function
# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_  # model 1
# oversimplify the model
# g = x1_*w1_ + x3_*w3_ + x4_*w4_ # model 2
# g = x1_*w1_ + x4_*w4_ # model 3
# overcomplicate the model
# x1_sq = x1**2
# x2_sq = x2**2
# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_ + x1_sq*w5_ # model 4
# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_ + x1_sq*w5_ + x2_sq_*w6_ # model 5


# end of the code ------------

# define the gradient descent optimizator for multivariate linear regression problem
# ------------ your code goes here



# end of the code ------------

# model 2
# train the models, for each of them you should get the model's rmse and parameter values (compare them with the true parameter values)
# ------------ your code goes here
# model 1
# model 3
# model 4
# model 5
# end of the code ------------

# we will discuss the final, model validation part, during the upcoming workshops