import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle

data = pd.read_csv("fruits.txt", sep="\t")
# print(data.head())

# encoding words to integer values
le = preprocessing.LabelEncoder()
fruit_name = le.fit_transform(list(data["fruit_name"]))

# features
x = data[["mass", "height", "width"]]
# labels
y = list(fruit_name)

# print(x)
# print(y)

# finding the best accuracy for the model and saving it with pickle
'''best = 0
for _ in range(100):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    if acc > best:
        best = acc
        print(best)
        with open("challenge.pickle", "wb") as f:
            pickle.dump(model, f)'''

# opening the model
fh = open("challenge.pickle", "rb")
model = pickle.load(fh)

# with the input, the program predicts the fruit
inp1 = input("Enter a mass: ")
inp2 = input("Enter a height: ")
inp3 = input("Enter a width: ")

inp = [[inp1, inp2, inp3]]

# inp = [[100, 6.3, 8]]
name = ["apple", "mandarin", "orange", "lemon"]
new_output = model.predict(inp)

print(name[new_output[0]])


