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