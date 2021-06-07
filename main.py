import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict
import random

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
    print("Ejercicio #1 con version original de los datos.")
    print("Ordinary least squares:")
    ordinary_least_squares(X, y, X2, y2, True)
    print("Lasso regression:")
    lasso_regression(X, y, X2, y2, True)


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

    print("\n\n")
    print("Ejercicio 1 con Datos normalizados: ")
    print("Ordinary least squares:")
    ordinary_least_squares(X, y, X2, y2, True)
    print("Lasso regression:")
    lasso_regression(X, y, X2, y2, True)




def ordinary_least_squares(X, y, X2, y2, will_print):
    reg = linear_model.LinearRegression().fit(X, y)
    y_pred = reg.predict(X2)

    if will_print:
        # print("Prediction:")
        # print(y_pred)
        print("Reg.coef:")
        print(reg.coef_)
        print("Score")
        print(reg.score(X2, y2))
        print("MSE")
        print(mean_squared_error(y2, y_pred))
        print("\n")

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
        print("\n")

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
    # print("Lo que le mandamos:")
    # print("X")
    # print(X)
    # print("X2")
    # print(X2)
    print("\n\n")

    regresion_logistica(X, y, X2, y2, True)

def regresion_logistica(X, y, X2, y2, will_print):
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict(X2)

    if will_print:
        # print("Prediction:")
        # print(y_pred)
        #print(confusion_matrix(y2, y_pred))
        print(classification_report(y2, y_pred))
        #print(precision_score())
        print('Coeficientes')
        print(clf.coef_)



#Ejercicio 3
def Setup_random_forests():
    #Initialize Table
    TableArrayStuff = []
    current_ID = 0

    #Set Columns for table
    columns = ["Conf. ID","Criterion","N_Estimators","Max_Depth","Max_Features", "P1","P2","P3","P4","P5","Promedio"]

    # Import dataset using Pandas
    dataset = pd.read_csv("genero_peliculas_training.csv")
    test_dataset = pd.read_csv("genero_peliculas_testing.csv")

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

    criterion_list = ["gini", "entropy"]
    n_estimators_list = [90, 100, 150, 200]
    max_depth_list = [None, 5, 10, 3]
    max_features_list = ["auto", "sqrt", "log2"]

    # for i in range(16):
    #     TableArrayStuff.append(random_forests(X, y, X2, y2, True, current_ID, criterion_list[random.randint(0, 1)], n_estimators_list[random.randint(0, 3)], max_depth_list[random.randint(0, 3)], max_features_list[random.randint(0, 2)]))
    #     current_ID = current_ID + 1

    for criterion in criterion_list:
       for n_estimators in n_estimators_list:
           for max_depth in max_depth_list:
               for max_features in max_features_list:
                   #TableArrayStuff.append(random_forests(X, y, X2, y2, True, current_ID, criterion, n_estimators, max_depth, max_features))
                   current_ID = current_ID + 1

    # print("Resulting Table. Not ordered.")
    # Table_dataframe = pd.DataFrame(np.array(TableArrayStuff),columns=columns)
    # print(Table_dataframe)
    # Table_dataframe.to_csv("results_ejercicio3.csv")

    print("Resultados de Ejercicio 3 Parte 1 fueron guardados en results_ejercicio3.csv.")
    criterion = "entropy"
    n_estimators = 100
    max_depth = 3
    max_features = "sqrt"

    random_forests_part2(X, y, X2, y2, True)

    random_forests_part3(X,y, X2, y2, True)



def random_forests(X, y, X2, y2, will_print,current_ID, criterion, n_estimators, max_depth, max_features):
    depth = ""
    if max_depth is None:
        depth = "None"
    else:
        depth = max_depth

    clf = ensemble.RandomForestClassifier( criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features ).fit(X,y)
    cv = cross_validate(clf, X, y, cv=5)
    print(current_ID)
    # print('CV')
    # print(cv)
    #lista de scores
    # print(cv['test_score'])
    # print(cv['test_score'].mean())
    Tuple = [current_ID, criterion, n_estimators, depth, max_features]
    for thing in cv['test_score']:
        Tuple.append(thing)
    Tuple.append(cv['test_score'].mean())
    return Tuple


def random_forests_part2(X, y, X2, y2, will_print):
    total_matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    confusion_matrixes = []
    model = ensemble.RandomForestClassifier(criterion="entropy", n_estimators=100, max_depth=3,max_features="sqrt")
    kf = KFold(n_splits=5)
    print("Ejercicio 3 Parte 2")
    for train_index, test_index in kf.split(X):

        # print("train index")
        # print(train_index)
        # print("test index")
        # print(test_index)
        X = np.array(X)
        y = np.array(y)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if will_print:
            print("Confusion")
            print(confusion_matrix(y_test, y_pred))
            print("Accuracy")
            print(accuracy_score(y_test, y_pred))
            confusion_matrixes.append(confusion_matrix(y_test, y_pred))
    for matrix in confusion_matrixes:
        for i in range(4):
            for j in range(4):
                total_matrix[i][j] += matrix[i][j]
    print("Confusion totals")
    print(total_matrix)
    total_datos = 0
    for i in range(4):
        for j in range(4):
            total_datos += total_matrix[i][j]

    #accion
    accion = ["accion"]
    precision = 0
    for i in range(4):
        precision += total_matrix[i][0]
    precision = total_matrix[0][0]/precision
    accion.append(precision)
    recall = 0
    for i in range(4):
        recall += total_matrix[0][i]
    precision = total_matrix[0][0] / recall
    accion.append(recall)
    accuracy = 0
    for i in range(4):
        for j in range(4):
            if(i != 0 and j != 0):
                accuracy += total_matrix[i][j]
    accuracy = accuracy/total_datos
    accion.append(accuracy)
    f = 2*precision*recall/(precision+recall)
    accion.append(f)

    #comedia
    comedia = ["comedia"]
    precision = 0
    for i in range(4):
        precision += total_matrix[i][1]
    precision = total_matrix[1][1] / precision
    comedia.append(precision)
    recall = 0
    for i in range(4):
        recall += total_matrix[1][i]
    precision = total_matrix[1][1] / recall
    comedia.append(recall)
    accuracy = 0
    for i in range(4):
        for j in range(4):
            if (i != 1 and j != 1):
                accuracy += total_matrix[i][j]
    accuracy = accuracy / total_datos
    comedia.append(accuracy)
    f = 2 * precision * recall / (precision + recall)
    comedia.append(f)

    #drama
    drama = ["drama"]
    precision = 0
    for i in range(4):
        precision += total_matrix[i][2]
    precision = total_matrix[2][2] / precision
    drama.append(precision)
    recall = 0
    for i in range(4):
        recall += total_matrix[2][i]
    precision = total_matrix[2][2] / recall
    drama.append(recall)
    accuracy = 0
    for i in range(4):
        for j in range(4):
            if (i != 2 and j != 2):
                accuracy += total_matrix[i][j]
    accuracy = accuracy / total_datos
    drama.append(accuracy)
    f = 2 * precision * recall / (precision + recall)
    drama.append(f)

    horror = ["horror"]
    precision = 0
    for i in range(4):
        precision += total_matrix[i][3]
    precision = total_matrix[3][3] / precision
    horror.append(precision)
    recall = 0
    for i in range(4):
        recall += total_matrix[3][i]
    precision = total_matrix[3][3] / recall
    horror.append(recall)
    accuracy = 0
    for i in range(4):
        for j in range(4):
            if (i != 3 and j != 3):
                accuracy += total_matrix[i][j]
    accuracy = accuracy / total_datos
    horror.append(accuracy)
    f = 2 * precision * recall / (precision + recall)
    horror.append(f)
    print("genero, precision, recall, accuracy, f1")
    print(accion)
    print(comedia)
    print(drama)
    print(horror)


def random_forests_part3(X,y, X2, y2, will_print):
    model = ensemble.RandomForestClassifier(criterion="entropy", n_estimators=100, max_depth=3, max_features="sqrt").fit(X,y)
    y_pred = model.predict(X2)

    print("Ejercicio 3 Parte 3")
    print(confusion_matrix(y2,y_pred))
    print(classification_report(y2, y_pred))

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