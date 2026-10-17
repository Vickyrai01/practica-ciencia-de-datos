import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


candidatos = {
    "Logistic Regression": (
        Pipeline([("tfidf", TfidfVectorizer()),
                  ("classifier", LogisticRegression(max_iter=1000, random_state=42))]),
        {"classifier__C": [0.01, 0.1, 1, 10, 100]}
    ),
    "Multinomial Naive Bayes": (
        Pipeline([("tfidf", TfidfVectorizer()),
                  ("classifier", MultinomialNB())]),
        {"classifier__alpha": [0.01, 0.1, 1, 10, 100]}
    ),
    "Decision Tree": (
        Pipeline([("tfidf", TfidfVectorizer()),
                  ("classifier", DecisionTreeClassifier(random_state=42))]),
        {"classifier__max_depth": [3, 5, 10, None]}
    ),
    "Random Forest": (
        Pipeline([("tfidf", TfidfVectorizer()),
                  ("classifier", RandomForestClassifier(random_state=42))]),
        {"classifier__n_estimators": [50, 100],
         "classifier__max_depth": [5, 10, None]}
    ),
    "SVM lineal": (
        Pipeline([("tfidf", TfidfVectorizer()),
                  ("classifier", SVC(kernel="linear"))]),
        {"classifier__C": [0.01, 0.1, 1, 10, 100]}
    ),
}

# Funcion para seleccionar la mejor configuracion de un modelo
def seleccionar_modelo(nombre, pipeline, param_grid, X_train, y_train):
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    idx = grid.best_index_
    resultado = {
        "Modelo": nombre,
        "Macro F1 CV": grid.best_score_,
        "Desvío entre folds": grid.cv_results_["std_test_score"][idx],
        "Hiperparámetros": grid.best_params_
    }
    return grid, resultado

# Ejecutar SOLO después de conectar X_train e y_train:
# grids = {}
# filas = []
# for nombre, (pipeline, parametros) in candidatos.items():
#     grid, fila = seleccionar_modelo(nombre, pipeline, parametros, X_train, y_train)
#     grids[nombre] = grid
#     filas.append(fila)
# comparacion = pd.DataFrame(filas).sort_values("Macro F1 CV", ascending=False)
# display(comparacion)


def graficar_learning_curve(nombre, modelo, X_train, y_train):
    tamanios, train_scores, val_scores = learning_curve(
        modelo, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5)
    )
    plt.figure(figsize=(8, 5))
    plt.plot(tamanios, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    plt.plot(tamanios, val_scores.mean(axis=1), marker="o", label="Validación")
    plt.xlabel("Cantidad de ejemplos de entrenamiento")
    plt.ylabel("Macro F1")
    plt.title(f"Learning Curve — {nombre}")
    plt.legend()
    plt.show()


def graficar_validation_curve(nombre, modelo, X_train, y_train,
                              parametro, valores, escala_log=False):
    train_scores, val_scores = validation_curve(
        modelo, X_train, y_train, param_name=parametro,
        param_range=valores, cv=5, scoring="f1_macro", n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    plt.plot(valores, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    plt.plot(valores, val_scores.mean(axis=1), marker="o", label="Validación")
    if escala_log:
        plt.xscale("log")
    plt.xlabel(parametro.split("__")[-1])
    plt.ylabel("Macro F1")
    plt.title(f"Validation Curve — {nombre}")
    plt.legend()
    plt.show()

# Ejemplo, DESPUÉS de crear grids:
# finalista = "Logistic Regression"
# graficar_learning_curve(finalista, grids[finalista].best_estimator_, X_train, y_train)
# graficar_validation_curve(finalista, grids[finalista].best_estimator_,
#                           X_train, y_train, "classifier__C", [0.01, 0.1, 1, 10, 100], True)


## funcion para evaluar todos los modelos en base a las metricas que decidimos optimizar
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

def evaluar_modelos_cv(grids, configuraciones, X_train, y_train):
    """Diagnósticos de selección; NO recibe X_test ni y_test."""
    resumen = []
    for nombre, grid in grids.items():
        modelo = grid.best_estimator_
        config = configuraciones.get(nombre, {})
        # NUESTRA ADAPTACIÓN: predicciones fuera de fold para cada texto de entrenamiento.
        y_pred_cv = cross_val_predict(modelo, X_train, y_train, cv=5, n_jobs=-1)
        clases = sorted(pd.Series(y_train).unique())

        reales = pd.Series(y_train).value_counts(normalize=True).reindex(clases, fill_value=0)
        predichas = pd.Series(y_pred_cv).value_counts(normalize=True).reindex(clases, fill_value=0)
        diferencia = predichas - reales  # positivo = sobreestimación de la clase
        error_promedio_pp = diferencia.abs().mean() * 100
        # FIN DE NUESTRA ADAPTACIÓN de distribuciones.

        reporte = classification_report(y_train, y_pred_cv, output_dict=True, zero_division=0)
        train_sizes, train_scores, val_scores = learning_curve(
            modelo, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5)
        )
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{nombre} — diagnóstico en validación (sin test)', fontsize=15)

        # 1. Matriz de confusión OOF (el gráfico original de la cátedra, en multiclase)
        cm = confusion_matrix(y_train, y_pred_cv, labels=clases)
        axes[0, 0].imshow(cm, cmap='Blues')
        for i in range(len(clases)):
            for j in range(len(clases)):
                axes[0, 0].text(j, i, cm[i, j], ha='center', va='center')
        axes[0, 0].set(xticks=range(len(clases)), yticks=range(len(clases)),
                       xticklabels=clases, yticklabels=clases,
                       xlabel='Predicho', ylabel='Real', title='1. Matriz de confusión CV')
        axes[0, 0].tick_params(axis='x', labelrotation=30)

        # 2. F1 por clase del classification_report: adaptación de visualización
        f1_clases = [reporte[str(c)]['f1-score'] for c in clases]
        axes[0, 1].bar([str(c) for c in clases], f1_clases)
        axes[0, 1].set(title='2. F1 por clase (CV)', ylabel='F1', ylim=(0, 1.05))
        axes[0, 1].tick_params(axis='x', labelrotation=30)

        # 3. NUESTRA ADAPTACIÓN: distribución real/predicha (porcentajes)
        x = np.arange(len(clases))
        axes[0, 2].bar(x - 0.2, reales.values * 100, width=0.4, label='Real')
        axes[0, 2].bar(x + 0.2, predichas.values * 100, width=0.4, label='Predicha')
        axes[0, 2].set(xticks=x, xticklabels=clases,
                       title=f'3. Distribución CV (error medio: {error_promedio_pp:.1f} pp)',
                       ylabel='Porcentaje de textos')
        axes[0, 2].tick_params(axis='x', labelrotation=30)
        axes[0, 2].legend()
        # FIN DE NUESTRA ADAPTACIÓN de distribuciones.

        # 4. Learning curve (profesor, cambiando a X_train y Macro F1)
        axes[1, 0].plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Train')
        axes[1, 0].plot(train_sizes, val_scores.mean(axis=1), marker='o', label='Validación')
        axes[1, 0].set(title='4. Learning Curve', xlabel='Textos de entrenamiento', ylabel='Macro F1', ylim=(0,1.05))
        axes[1, 0].legend()

        # 5. Validation curve (profesor, solo si la configuración lo especifica)
        if 'parametro' in config and 'valores' in config:
            tc, vc = validation_curve(
                modelo, X_train, y_train, param_name=config['parametro'],
                param_range=config['valores'], cv=5, scoring='f1_macro', n_jobs=-1
            )
            axes[1, 1].plot(config['valores'], tc.mean(axis=1), marker='o', label='Train')
            axes[1, 1].plot(config['valores'], vc.mean(axis=1), marker='o', label='Validación')
            if config.get('escala_log', False):
                axes[1, 1].set_xscale('log')
            axes[1, 1].set(title='5. Validation Curve', xlabel=config['parametro'], ylabel='Macro F1', ylim=(0,1.05))
            axes[1, 1].legend()
        else:
            axes[1, 1].set_title('5. Sin hiperparámetro para curva')
            axes[1, 1].axis('off')

        # 6. Puntaje en cada fold del GridSearch (diagnóstico de variabilidad)
        indice = grid.best_index_
        valores_fold = [grid.cv_results_[f'split{i}_test_score'][indice] for i in range(grid.n_splits_)]
        axes[1, 2].bar(range(1, len(valores_fold) + 1), valores_fold)
        axes[1, 2].set(title='6. Macro F1 por fold del GridSearch', xlabel='Fold', ylabel='Macro F1', ylim=(0,1.05))

        plt.tight_layout()
        plt.show()
        print('Modelo:', nombre)
        print('Mejores hiperparámetros:', grid.best_params_)
        print('Macro F1 CV (selección):', round(grid.best_score_, 4))
        print('Diferencia media de distribución:', round(error_promedio_pp, 2), 'pp')
        print('Reporte de predicciones de CV (diagnóstico):')
        print(classification_report(y_train, y_pred_cv, zero_division=0))
        resumen.append({
            'Modelo': nombre,
            'Macro F1 CV (GridSearch)': grid.best_score_,
            'Error medio distribución (pp)': error_promedio_pp,
            'Mejores parámetros': grid.best_params_
        })
    return pd.DataFrame(resumen).sort_values('Macro F1 CV (GridSearch)', ascending=False)

# Ejecutar cuando ya estén definidos y entrenados estos cuatro grids.
# grids = {
#     'Logistic Regression': grid_lr,
#     'Naive Bayes': grid_nb,
#     'SVM': grid_svm,
#     'Random Forest': grid_rf
# }
# configuraciones = {
#     'Logistic Regression': {'parametro':'classifier__C','valores':[0.01,0.1,1,10,100],'escala_log':True},
#     'Naive Bayes': {'parametro':'classifier__alpha','valores':[0.01,0.1,1,10,100],'escala_log':True},
#     'SVM': {'parametro':'classifier__C','valores':[0.01,0.1,1,10,100],'escala_log':True},
#     'Random Forest': {'parametro':'classifier__n_estimators','valores':[50,100,150,200]}
# }
# tabla_comparativa = evaluar_modelos_cv(grids, configuraciones, X_train, y_train)
# tabla_comparativa

def evaluar_modelo_final(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print(f"Modelo final: {nombre}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=["N", "NEU", "P"], cmap="Blues"
    )
    plt.title(f"Matriz de confusión — {nombre}")
    plt.show()
    return y_pred

# Ejecutar AL FINAL, cuando la selección esté cerrada:
# predicciones = evaluar_modelo_final(nombre_elegido, modelo_elegido, X_test, y_test)
