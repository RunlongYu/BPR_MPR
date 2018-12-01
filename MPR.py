from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import scores

class MPR_SS:
    user_count = 943
    item_count = 1682
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    lambda_mpr = 0.7
    train_count = 1000
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    first_flag = 1
    iteration_interval = int(item_count * item_count)
    factor_ranking = np.random.rand(latent_factors, item_count)
    ranking_pro = np.random.rand(item_count)
    wheel = np.zeros(item_count)
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01
    V = np.random.rand(item_count, latent_factors) * 0.01
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            # sample positive items from the observed items
            if len(user_ratings_train[u]) > 2:
                i, p, pp = random.sample(user_ratings_train[u], 3)
            else:
                i = random.sample(user_ratings_train[u], 1)[0]
                p = random.sample(user_ratings_train[u], 1)[0]
                pp = random.sample(user_ratings_train[u], 1)[0]
            # sample negative items
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            qq = random.randint(1, self.item_count)
            while qq in user_ratings_train[u]:
                qq = random.randint(1, self.item_count)
            # adaptive sampler
            q = int(self.sampling_Strategy(u))
            while q in user_ratings_train[u]:
                q = int(self.sampling_Strategy(u))
            u -= 1
            i -= 1
            pp -= 1
            p -= 1
            j -= 1
            q -= 1
            qq -= 1
            r_ui = np.dot(self.U[u], self.V[i].T)
            r_up = np.dot(self.U[u], self.V[i].T)
            r_upp = np.dot(self.U[u], self.V[i].T)
            r_uj = np.dot(self.U[u], self.V[j].T)
            r_uq = np.dot(self.U[u], self.V[j].T)
            r_uqq = np.dot(self.U[u], self.V[j].T)
            x = self.lambda_mpr * (r_ui - r_uj - r_uq + r_uqq) + (1 - self.lambda_mpr) * (r_uq - r_uqq - r_up + r_upp)
            mid = 1.0 / (1 + np.exp(x))
            temp = self.U[u]
            self.U[u] += -self.lr * (-mid * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (-mid * temp + self.reg * self.V[i])
            self.V[j] += -self.lr * (-mid * (-temp) + self.reg * self.V[j])

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def set_up(self):
        sum = 0
        for i in range(self.item_count):
            self.ranking_pro[i] = np.exp(-i / (np.log(self.user_count) * np.log(self.item_count)) - 1.0)
            sum += self.ranking_pro[i]
        self.wheel[0] = self.ranking_pro[0] / sum
        for i in range(self.item_count - 1):
            self.ranking_pro[i + 1] = self.ranking_pro[i + 1] / sum
            self.wheel[i + 1] = self.wheel[i] + self.ranking_pro[i + 1]

    def geometrics(self):
        r = random.uniform(0, 1)
        for i in range(self.item_count):
            if self.wheel[i] > r:
                return i

    def sampling_Strategy(self, u):
        u -= 1
        # draw a position from geometrics distribution
        position_r = self.geometrics()
        while position_r > self.item_count:
            position_r = self.geometrics()
        # sample a latent factor
        factorId = random.randint(0, self.latent_factors - 1)
        if self.U[u][factorId] > 0:
            return self.factor_ranking[factorId][int(position_r)]
        else:
            return self.factor_ranking[factorId][int(position_r)]

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        for i in range(self.user_count * self.item_count):
            self.test[i] = int(self.test[i])
        # training
        for i in range(self.train_count):
            if i % self.iteration_interval == 0:
                self.set_up()
            self.train(user_ratings_train)
        predict_matrix = self.predict(self.U, self.V)
        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('auc:', auc_score)
        # Top-K evaluation
        str(scores.topK_scores(self.test, self.predict_, 20, self.user_count, self.item_count))

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u-1) * item_count + j - 1]= 0
    return predict

if __name__ == '__main__':
    mpr_ss = MPR_SS()
    mpr_ss.main()