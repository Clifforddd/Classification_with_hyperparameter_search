#pip install numpy scikit-learn

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def collect_data(num_points, prompt):
    data = []
    for _ in range(num_points):
        x = float(input(f"Enter {prompt} x value: "))
        y = int(input(f"Enter {prompt} y value: "))
        data.append((x, y))
    return data

def main():
    N = int(input("Enter number of training data points (N): "))
    train_data = collect_data(N, "training")

    M = int(input("Enter number of test data points (M): "))
    test_data = collect_data(M, "test")

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    X_train, y_train = train_data[:, 0], train_data[:, 1]
    X_test, y_test = test_data[:, 0], test_data[:, 1]

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    parameters = {'n_neighbors': range(1, 11)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, cv=5)
    clf.fit(X_train, y_train)

    best_k = clf.best_params_['n_neighbors']
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    predictions = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"The best k is {best_k} with test accuracy of {accuracy:.2f}")

if __name__ == "__main__":
    main()
