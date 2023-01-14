import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

class DNBC(BaseEstimator, ClassifierMixin):
    def __init__(self, sizes, lp=False, lg=False):
        self.sizes_ = sizes
        self.labels_ = None
        self.Pt_ = None
        self.Pj_ = None

        self.lp_ = lp
        self.lg_ = lg
#------------------------
    def fit(self, X, y):
        self.labels_ = np.unique(y)
        r, c = X.shape
        ylabs = np.zeros(r, np.int8)
        
        for index, label in enumerate(self.labels_):
            ixs = y == label
            ylabs[ixs] = index

        self.Pt_ = np.zeros(self.labels_.size)
        for k in range(self.labels_.size):
            self.Pt_[k] = np.sum(ylabs == k) / r

        self.Pj_ = np.zeros((self.labels_.size, c, np.max(self.sizes_)))
        for i in range(r):
            for j in range(c):
                v = X[i, j]
                self.Pj_[ylabs[i], j][v] += 1

        for k in range(self.labels_.size):
            sum = self.Pt_[k] * r
            if self.lp_: # --- Laplace = True/False ---
                for j in range(c):
                    self.Pj_[k, j] = (self.Pj_[k, j] + 1) / (sum + self.sizes_[j])
            else:
                self.Pj_[k] /= self.Pt_[k] * r
#------------------------
    def predict(self, X):
        return self.labels_[np.argmax(self.predict_proba(X), axis=1)]
#------------------------
    def predict_proba(self, X):
        r, c = X.shape
        scores = np.ones((r, self.labels_.size))
        for x in range(r):
            for y in range(self.labels_.size):
                if self.lg_:
                    scores[x, y] = np.log(self.Pt_[y])
                else:
                    scores[x, y] = self.Pt_[y]
                for z in range(c):
                    if self.lg_:
                        scores[x, y] += np.log(self.Pj_[y, z, X[x, z]])
                    else:
                        scores[x, y] *= self.Pj_[y, z, X[x, z]]
        return scores

# Data *
Dataw = np.genfromtxt("wine.data", delimiter=",")
y = Dataw[:, 0].astype(np.int8) #
X = Dataw[:, 1:]                # - 

Datawf = np.genfromtxt("waveform.data", delimiter=",")
yy = Datawf[:, -1].astype(np.int8) #
XX = Datawf[:, :-1]                #

BINS = 3 #



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, train_size=0.30, random_state=50)


est = KBinsDiscretizer(BINS, encode='ordinal', strategy='uniform')
est.fit(X_train)
est.fit(X_test)
X_d_train = np.round(est.transform(X_train)).astype(int)
X_d_test = np.round(est.transform(X_test)).astype(int)


r, c = X_train.shape
max = np.max(X_train)
sizes = np.round(np.ones(c) * (max)).astype(int)

print("Data: WINE --- Bins: 3 --- Laplace: NO --- Logarithms: NO")
print("===================================================================================")
print("Dataset have", r, "rows and", c, "columns.")
Classify = DNBC(sizes, lp=False, lg=False)
Classify.fit(X_d_train, y_train)
print("Classifier:", Classify.Pt_)

predictions = Classify.predict(X_d_train)
accuracy_train = np.mean(y_train == predictions)
print("Learning accuracy: " + str(accuracy_train))
predictions = Classify.predict(X_d_test)
accuracy_test = np.mean(y_test == predictions)
print("Test accuracy: " + str(accuracy_test))

print("===================================================================================")