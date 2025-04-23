from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target, iris.target_names


def prepare_data(X, y, target_names, test_size=0.21, random_state=31):
    return (
        train_test_split(X, y, test_size=test_size, random_state=random_state),
        target_names,
    )
