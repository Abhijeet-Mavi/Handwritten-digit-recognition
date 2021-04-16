import numpy as np
from sklearn.datasets import fetch_openml
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


mnist = fetch_openml("mnist_784")
x, y = mnist['data'], mnist['target']
# print(x.shape)
# print(y.shape)

some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation='nearest')
plt.show()

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)

# print(y_train)

clf = LogisticRegression(tol = 0.1)

clf.predict([some_digit])

a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean)