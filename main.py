import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model




def Setup_regresion_lineal():
    dataset = pd.read_csv("regression_train.csv")
    #my_data = genfromtxt('regression_train.csv', delimiter=',')
    #print(my_data)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, 5]

    ordinary_least_squares(X, y)
    '''
    #Version Normalizada
    scaler = StandardScaler()
    scaler.fit(X)
    #print(scaler.mean_)
    #print(scaler.transform(X))
    '''


def ordinary_least_squares(dataset, target):
    reg = linear_model.LinearRegression()
    reg.fit(dataset, target)
    #print(reg.coef_)

def lasso_regression(dataset, target):
    reg = linear_model.LassoLars(alpha=.1)
    reg.fit(dataset, target)

ContinueOnMenu = True
while ContinueOnMenu:
    print("*** Menu ***")
    print("1. Regresion Lineal.")
    print("2. Regresion Logistica.")
    print("3. Random Forests.")
    selection = int(input("Ingrese una opcion [1-3]: "))
    if selection == 1:

        Setup_regresion_lineal()
