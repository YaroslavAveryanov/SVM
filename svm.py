import numpy as np
import pandas as pd
import cvxopt

#kernel = 'linear', 'rdf', 'poly'
#labels_y have to be -1 or +1
class Svm(object):
    def __init__(self, C=1.0, kernel = 'linear', gamma = 0.5, coef_pol = 0.0, degree = 2):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef_pol = coef_pol
        self.degree = degree
    
    def train(self, X, y):    
        K = self.gram_matrix(X)
        n = y.shape[0]
        
        #Matrices for quadratic program
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-np.ones(n))
        
        #equality constarins
        G1 = np.diag(-np.ones(n))
        h1 = np.zeros((n,1))
        
        G2 = np.diag(np.ones(n))
        h2 = self.C * np.ones((n,1))
        
        G = cvxopt.matrix(np.vstack((G1, G2)))
        h = cvxopt.matrix(np.vstack((h1,h2)))
        
        #inequality constrains (only one inequality)
        A = cvxopt.matrix(y, (1,n), tc='d')
        b = cvxopt.matrix(0.0)
        
        lambdas = cvxopt.solvers.qp(P, q, G, h, A, b)
        lambdas = np.squeeze(np.round(np.array(lambdas['x']), decimals=15))
        
        if self.kernel == 'poly':
            self.support_indexes = lambdas > 10**-12
        else:
            self.support_indexes = lambdas > 10**-4
        self.lambdas = lambdas[self.support_indexes]
        self.support_vectors = X[self.support_indexes,:]
        self.sup_vec_labels = y[self.support_indexes]
        self.coef = self.get_weights()
        self.bias = self.compute_bias()
    
    def compute_bias(self):
        if self.C >= 1:
            indexes = (self.lambdas > 0.1) & (self.lambdas < 0.8 * self.C)
        else:
            indexes = (self.lambdas > 0.1 * self.C) & (self.lambdas < 0.8 * self.C)
        
        if np.all(self.lambdas <= 0.1* self.C):
            indexes = self.lambdas < 0.1* self.C
        
       
        max_index = np.argmax(self.lambdas[indexes])
        
        sup = self.support_vectors[indexes][max_index]
        label = self.sup_vec_labels[indexes][max_index]
        
        total = -label
        for i in np.arange(self.lambdas.shape[0]):
            total += self.lambdas[i] * self.sup_vec_labels[i] * self.f_kernel(sup, self.support_vectors[i])
        
        return -total
    
    def gram_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n,n))
        for i in np.arange(n):
            for j in np.arange(n):
                K[i,j] = self.f_kernel(X[i,:], X[j,:])
        
        return K
    
    def f_kernel(self, a, b):
        
        if self.kernel == 'linear':
            return np.dot(a,b)
        elif self.kernel == 'rbf':
            return np.exp(-la.norm(a-b)**2 * self.gamma)
        elif self.kernel == 'poly':
            return (np.dot(a,b) + self.coef_pol) ** self.degree
        else:
            return -0.013
        
    def predict(self, x):
        res = self.bias
        for lambda_i, y_i, x_i in zip(self.lambdas, self.sup_vec_labels, self.support_vectors):
            res += lambda_i * y_i * self.f_kernel(x_i, x)
        
        return np.sign(res)
    
    def get_weights(self):
        n = self.support_vectors.shape[0]
        weights = np.zeros(self.support_vectors.shape[1])
        for i in np.arange(n):
            weights += self.lambdas[i] * self.sup_vec_labels[i] * self.support_vectors[i,:]
            
        return weights
    
    def pred_margin(self,x):
        res = self.bias
        for lambda_i, y_i, x_i in zip(self.lambdas, self.sup_vec_labels, self.support_vectors):
            res += lambda_i * y_i * self.f_kernel(x_i, x)
        return (res, np.sign(res))

import random
class Multi_Svm(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.clfs = []
        
    def train(self, X, y, gamma = 0.6, C = 10):
        for k1 in np.arange(self.n_classes):
            for k2 in np.arange(k1+1,self.n_classes):
                print 'k1 = ', k1, ', k2 = ', k2
                data_k = self.data_one_vs_one(k1, k2, X, y)
                y_k = data_k[0]
                X_k = data_k[1]

                clf = Svm(kernel='poly', gamma=0.6, C=10, degree=2)
                clf.train(X_k, y_k)
                self.clfs.append([clf, k1, k2])
    
    def data_one_vs_one(self, k1, k2, X_train, y_train):
        indexes_k1 = (y_train == k1)
        indexes_k2 = (y_train == k2)
        y_train_k = np.concatenate((y_train[indexes_k1], y_train[indexes_k2]))
        y_train_k = self.one_vs_one_transformed_labels(k1,k2,y_train_k)
        X_train_k = np.vstack((X_train[indexes_k1], X_train[indexes_k2]))
        return y_train_k, X_train_k
    
    def one_vs_one_transformed_labels(self, k1, k2, y_train_k):
        y = np.zeros(y_train_k.shape[0])
        for i in np.arange(y_train_k.shape[0]):
            if y_train_k[i] == k1:
                y[i] = 1
            else:
                y[i] = -1
        return y 
    
    def predict(self, X):
        predictions = []
        size = X.shape[0]

        for j in np.arange(size):
            x = X[j,:]
            scores = np.zeros(self.n_classes)
            for i in np.arange(len(self.clfs)):
                temp = self.clfs[i]
                clf = temp[0]
                k1 = temp[1]
                k2 = temp[2]
                pred = clf.predict(x)
                if pred == 1: 
                    scores[k1] += 1 
                else: 
                    scores[k2] += 1
            predictions.append(np.random.choice(np.where(scores==max(scores))[0]))

            if j % 100 == 0:
                print j    

        return np.array(predictions)

class k_means:
    def __init__(self, k = 3, n_init = 5, max_iter = 5):
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        
    def train(self, X):
        all_J = []
        all_centroids = []
        all_r_nk = []
        for initialization in np.arange(self.n_init):
            r_nk = np.zeros((X.shape[0], self.k))
            centroids = X[random.sample(np.arange(1,X.shape[0]),self.k)]
            
            for iteration in np.arange(self.max_iter):
                for n in np.arange(X.shape[0]):
                    distances  = np.linalg.norm(X[n,:] - centroids, ord=2, axis=1)
                    r_nk[n, np.argmin(distances)] = 1 
                for i in np.arange(self.k):
                    centroids[i,:] = (np.sum(X * r_nk[:,i].reshape(X.shape[0],1), axis = 0)) / float(sum(r_nk[:,i]))
            J = 0
            for i in np.arange(X.shape[0]):
                for j in np.arange(self.k):
                    J += r_nk[i,j] * np.linalg.norm(X[i,:] - centroids[j,:], ord=2)
            
            all_J.append(J)
            all_centroids.append(centroids)
        
        all_J = np.array(all_J)
        all_centroids = np.array(all_centroids)
        
        index = np.argmin(all_J)
        
        self.centroids = all_centroids[index]
        self.J = all_J[index]
    
    def predict(self, X):
        n = X.shape[0]
        labels = []
        
        for i in np.arange(n):
            distances  = np.linalg.norm(X[i,:] - self.centroids, ord=2, axis=1)
            labels.append(np.argmin(distances))
        
        return labels
    
    def predict_soft(self, X):
        n = X.shape[0]
        labels = []
        
        for i in np.arange(n):
            z = np.linalg.norm(X[i,:]-self.centroids, ord=2, axis = 1)
            mu = np.mean(z)
            labels.append(np.maximum(mu - z, 0))
        
        return labels

def make_submission(y_predict, file_name):
    file_obj = open(file_name, 'w')
    string = "Id,Prediction\n"
    file_obj.write(string)
    
    i = 1
    for digit in y_predict:
        string = str(i) + ',' + str(digit) + '\n'
        file_obj.write(string)
        i += 1
    
    file_obj.close()

def image_representation(images, w, clf):
    X_image_repres = []

    for img in images:
        img_repres = np.zeros([(32-w+1), (32-w+1), clf.k])
        for i in np.arange(0,32-w):
            for j in np.arange(32-w):
                patch = img[i:i+w,j:j+w,:]
                k = clf.predict(patch.reshape(1, w*w*3))
                k_vector = np.zeros(clf.k)
                k_vector[k[0]] = 1
                img_repres[i,j,:] = k_vector
	    
        quad1 = np.sum(img_repres[:13,:13,:], axis=(0,1))
        quad2 = np.sum(img_repres[:13,13:,:], axis=(0,1))
        quad3 = np.sum(img_repres[13:,:13,:], axis=(0,1))
        quad4 = np.sum(img_repres[13:,13:,:], axis=(0,1))
        
        final_vector = np.concatenate((quad1, quad2, quad3, quad4))
        X_image_repres.append(final_vector)    
    
    return np.array(X_image_repres)
    

data = pd.read_csv('Xtr.csv', header = None)
data.drop(3072, axis=1, inplace=True)
labels = pd.read_csv("Ytr.csv")
y_train = labels.ix[:,1].as_matrix()
X_train = data.as_matrix()

data = pd.read_csv('Xte.csv', header = None)
data.drop(3072, axis=1, inplace=True)
X_test = data.as_matrix()

images_train = []
for i in range(X_train.shape[0]):
    images_train.append(X_train[i,:].reshape(3, 32, 32).transpose(1, 2, 0))
images_train = np.array(images_train)

images_test = []
for i in range(X_test.shape[0]):
    images_test.append(X_test[i,:].reshape(3, 32, 32).transpose(1, 2, 0))
images_test = np.array(images_test)

w = 6

patches = []
for i in np.arange(int(X_train.shape[0])):
    for j in np.arange(10):
        a = np.random.randint(0,32-w)
        b = np.random.randint(0,32-w)
        patch = images_train[i][a:a+w,b:b+w,:]
        patches.append(patch)
patches = np.array(patches)

k_means_train = patches.reshape(patches.shape[0], w*w*3)

K = 800
clf = k_means(K)
clf.train(k_means_train)

X_train_image_repres = image_representation(images_train, w, clf)
X_test_image_repres = image_representation(images_test, w, clf)

my_svm = Multi_Svm(n_classes=10)
my_svm.train(X_train_image_repres, y_train)

predictions = my_svm.predict(X_test_image_repres)

make_submission(predictions, 'Yte.csv')



