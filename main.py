import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#Ejercicio #1

def Setup_regresion_lineal():
    #Import dataset using Pandas
    dataset = pd.read_csv("regression_train.csv")
    test_dataset = pd.read_csv("regression_test.csv")

    #Set-up the X(training info) and y(training target) arrays.
    X = dataset.iloc[:, :-1]#try.values
    y = dataset.iloc[:, 5]

    #Set-up the X2(test info) and y2(training target) arrays.
    X2 = test_dataset.iloc[:, :-1]
    y2 = test_dataset.iloc[:, 5]

    # # Run the algorithms.
    # print("Lo que le mandamos:")
    # print(X)
    # # print("X2")
    # # print(X2)
    # print("\n\n")

    #ordinary_least_squares(X, y, X2, y2, True)
    #lasso_regression(X, y, X2, y2, True)


    #Setup de Normalizacion de Datos para la segunda corrida.

    #Requieres numpy array, if not prints the same array. Doesnt do anything. <--- Suspicious, NO LO HIZO.
    #Version Normalizada
    scaler = StandardScaler()
    scaler.fit(X)
    #print(scaler.mean_)
    X = scaler.transform(X)
    # print("X")
    # print(X)

    scaler = StandardScaler()
    scaler.fit(X2)
    #print(scaler.mean_)
    X2 = scaler.transform(X2)
    # print("X2")
    # print(X2)

    # # Run the algorithms.
    # print("Lo que le mandamos:")
    # print(X)
    # print("X2")
    # print(X2)
    # print("\n\n")

    ordinary_least_squares(X, y, X2, y2, True)
    lasso_regression(X, y, X2, y2, True)




def ordinary_least_squares(X, y, X2, y2, will_print):
    reg = linear_model.LinearRegression().fit(X, y)
    y_pred = reg.predict(X2)

    if will_print:
        print("Prediction:")
        print(y_pred)
        print("Reg.coef:")
        print(reg.coef_)
        print("Score")
        print(reg.score(X2, y2))
        print("MSE")
        print(mean_squared_error(y2, y_pred))


def lasso_regression(X, y, X2, y2, will_print):
    clf = linear_model.Lasso()
    clf.fit(X, y)
    y_pred = clf.predict(X2)

    if will_print:
        #print("Prediction:")
        #print(y_pred)

        print("Reg.coef:")
        print(clf.coef_)
        print("Intercepto")
        print(clf.intercept_)
        print("Score")
        print(clf.score(X2, y2))
        print("MSE")
        print(mean_squared_error(y2, y_pred))


#Ejercicio 2

def Setup_regresion_logistica():
    # Import dataset using Pandas
    dataset = pd.read_csv("seguros_training_data.csv")
    test_dataset = pd.read_csv("seguros_testing_data.csv")

    # Set-up the X(training info) and y(training target) arrays.
    X = dataset.iloc[:, :-1]  # try.values
    y = dataset.iloc[:, 10]

    # Binarization
    X = X.apply(LabelEncoder().fit_transform)

    # Set-up the X2(test info) and y2(training target) arrays.
    X2 = test_dataset.iloc[:, :-1]
    y2 = test_dataset.iloc[:, 10]

    # Binarization
    X2 = X2.apply(LabelEncoder().fit_transform)

    # Run the algorithms.
    print("Lo que le mandamos:")
    print("X")
    print(X)
    print("X2")
    print(X2)
    print("\n\n")

    regresion_logistica(X, y, X2, y2, True)

def regresion_logistica(X, y, X2, y2, will_print):
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict(X2)

    if will_print:
        print("Prediction:")
        print(y_pred)
        #print(confusion_matrix(y2, y_pred))
        print(classification_report(y2, y_pred))
        #print(precision_score())


#Ejercicio 3
def Setup_random_forests():
    # Import dataset using Pandas
    dataset = pd.read_csv("genero_peliculas_training.csv")
    test_dataset = pd.read_csv("genero_peliculas_testing.csv")

    # Set-up the X(training info) and y(training target) arrays.
    X = dataset.iloc[:, :-1]  # try.values
    y = dataset.iloc[:, 9]

    # Binarization
    X = X.apply(LabelEncoder().fit_transform)

    # Set-up the X2(test info) and y2(training target) arrays.
    X2 = test_dataset.iloc[:, :-1]
    y2 = test_dataset.iloc[:, 9]

    # Binarization
    X2 = X2.apply(LabelEncoder().fit_transform)

    # print(X)

    random_forests(X, y, X2, y2, True)



def random_forests(X, y, X2, y2, will_print):


    clf = ensemble.RandomForestClassifier(n_estimators=100).fit(X,y)
    y_pred = clf.predict(X2)
    print(y_pred)



#Menu
ContinueOnMenu = True
while ContinueOnMenu:
    print("*** Menu ***")
    print("1. Regresion Lineal.")
    print("2. Regresion Logistica.")
    print("3. Random Forests.")
    selection = int(input("Ingrese una opcion [1-3]: "))
    if selection == 1:
        Setup_regresion_lineal()
    elif selection == 2:
        Setup_regresion_logistica()
    elif selection == 3:
        Setup_random_forests()
    else:
        print("Unsupported")
        ContinueOnMenu = False





#Usefull trash

#Alleged Fix
    #labelEncoder_X = LabelEncoder()
    #X = X.apply(LabelEncoder().fit_transform)



#Alleged Fix
    #labelEncoder_X2 = LabelEncoder()
    #X2 = X2.apply(LabelEncoder().fit_transform)