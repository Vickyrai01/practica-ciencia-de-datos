# Función 1: seleccionar un modelo
from sklearn.model_selection import GridSearchCV

def seleccionar_modelo(nombre, pipeline, param_grid, X_train, y_train):
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(f"Modelo: {nombre}")
    print("Mejores hiperparámetros:", grid.best_params_)
    print("Mejor Macro F1 promedio:", grid.best_score_)

    return grid

# Función 2: evaluar el modelo seleccionado
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluar_modelo(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    print(f"Modelo: {nombre}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Matriz de confusión - {nombre}")
    plt.show()


# funcion para ver la curva de aprendizaje ¿Qué pasa cuando aumentamos la cantidad de datos?
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def graficar_learning_curve(nombre, modelo, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_sizes,
        train_scores.mean(axis=1),
        marker="o",
        label="Entrenamiento"
    )

    plt.plot(
        train_sizes,
        val_scores.mean(axis=1),
        marker="o",
        label="Validación"
    )

    plt.xlabel("Cantidad de textos de entrenamiento")
    plt.ylabel("Macro F1")
    plt.title(f"Learning Curve - {nombre}")
    plt.legend()
    plt.show()


# Funcion para ver la curva de validacion. ¿Qué pasa cuando modificamos la complejidad del modelo?
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

def graficar_validation_curve(
    nombre,
    modelo,
    X_train,
    y_train,
    parametro,
    valores
):
    train_scores, val_scores = validation_curve(
        modelo,
        X_train,
        y_train,
        param_name=parametro,
        param_range=valores,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        valores,
        train_scores.mean(axis=1),
        marker="o",
        label="Entrenamiento"
    )

    plt.plot(
        valores,
        val_scores.mean(axis=1),
        marker="o",
        label="Validación"
    )

    plt.xlabel(parametro.split("__")[-1])
    plt.ylabel("Macro F1")
    plt.title(f"Validation Curve - {nombre}")
    plt.legend()
    plt.show()