import os
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, 
                             precision_score, f1_score, matthews_corrcoef)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)

class OptimizationProcess:
    def __init__(self, df_data=None):
        self.df_data = df_data

    def generate_model_params(self, trial, algorithm):
        # Genera solo los parámetros específicos para cada modelo
        if algorithm == "KNeighborsClassifier":
            return {
                "n_neighbors": trial.suggest_int(f"{algorithm}_n_neighbors", 3, 30),
                "leaf_size": trial.suggest_int(f"{algorithm}_leaf_size", 3, 50),
                "algorithm": trial.suggest_categorical(f"{algorithm}_algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute']),
                "metric": trial.suggest_categorical(f"{algorithm}_metric", ["cityblock", "euclidean", "manhattan", "minkowski"])
            }
        elif algorithm == "DecisionTreeClassifier":
            return {
                "criterion": trial.suggest_categorical(f"{algorithm}_criterion", ["gini", "entropy", "log_loss"]),
                "splitter": trial.suggest_categorical(f"{algorithm}_splitter", ["best", "random"]),
                "min_weight_fraction_leaf": trial.suggest_float(f"{algorithm}_min_weight_fraction_leaf", 0, 0.5),
                "max_depth": trial.suggest_int(f"{algorithm}_max_depth", 1, 100),
                "min_impurity_decrease": trial.suggest_float(f"{algorithm}_min_impurity_decrease", 0, 100),
                "max_leaf_nodes": trial.suggest_int(f"{algorithm}_max_leaf_nodes", 2, 200)
            }
        elif algorithm == "RandomForestClassifier":
            return {
                "n_estimators": trial.suggest_int(f"{algorithm}_n_estimators", 10, 5000),
                "criterion": trial.suggest_categorical(f"{algorithm}_criterion", ["gini", "entropy", "log_loss"]),
                "min_samples_split": trial.suggest_int(f"{algorithm}_min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int(f"{algorithm}_min_samples_leaf", 1, 30),
                "max_features": trial.suggest_categorical(f"{algorithm}_max_features", ["sqrt", "log2"])
            }
        elif algorithm == "GradientBoostingClassifier":
            return {
                "loss": trial.suggest_categorical(f"{algorithm}_loss", ['log_loss', 'exponential']),
                "learning_rate": trial.suggest_float(f"{algorithm}_learning_rate", 0, 0.5),
                "n_estimators": trial.suggest_int(f"{algorithm}_n_estimators", 10, 5000),
                "criterion": trial.suggest_categorical(f"{algorithm}_criterion", ["friedman_mse", "squared_error"]),
                "min_samples_split": trial.suggest_int(f"{algorithm}_min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int(f"{algorithm}_min_samples_leaf", 1, 30),
                "max_depth": trial.suggest_int(f"{algorithm}_max_depth", 1, 10),
                "max_features": trial.suggest_categorical(f"{algorithm}_max_features", ["sqrt", "log2"]),
                "n_iter_no_change": trial.suggest_int(f"{algorithm}_n_iter_no_change", 1, 10)
            }
        elif algorithm == "SVC":
            return {
                "probability": True,
                "random_state": 42
            }
        elif algorithm == "ExtraTreesClassifier":
            return {
                "n_estimators": trial.suggest_int(f"{algorithm}_n_estimators", 10, 5000),
                "criterion": trial.suggest_categorical(f"{algorithm}_criterion", ["gini", "entropy", "log_loss"]),
                "min_samples_split": trial.suggest_int(f"{algorithm}_min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int(f"{algorithm}_min_samples_leaf", 1, 30),
                "max_features": trial.suggest_categorical(f"{algorithm}_max_features", ["sqrt", "log2"])
            }
        elif algorithm == "AdaBoostClassifier":
            return {
                "n_estimators": trial.suggest_int(f"{algorithm}_n_estimators", 10, 5000),
                "learning_rate": trial.suggest_float(f"{algorithm}_learning_rate", 0.01, 2.0),
                "algorithm": trial.suggest_categorical(f"{algorithm}_algorithm", ["SAMME", "SAMME.R"])
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def settings_models(self, trial, algorithm):
        # Obtiene los parámetros y crea el modelo
        params = self.generate_model_params(trial, algorithm)
        return globals()[algorithm](**params), params

    def objective_models(self, trial):
        model, params = self.settings_models(trial, self.algorithm)
        scores = cross_validate(model, self.X_train, self.y_train, cv=5, scoring="f1_weighted")
        trial.set_user_attr("params", params)  # Guardar parámetros en el trial
        return np.mean(scores["test_score"])

    def run_optimization(self, type_encoder=None, encoder=None, algorithm=None):
        self.algorithm = algorithm
        responses = self.df_data["monomer_state"].values
        df = self.df_data.drop(columns=["monomer_state"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df, responses, 
                                                                                random_state=42, test_size=0.3)
        output_path = f"../results/tuning/{type_encoder}/{encoder}"
        os.makedirs(output_path, exist_ok=True)

        print(f"Optimization {algorithm}")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_models, n_trials=50)

        best_params = study.best_trial.user_attrs["params"]
        model = globals()[algorithm](**best_params)
        model.fit(self.X_train, self.y_train)

        # Validación cruzada
        response_cv = cross_validate(model, self.X_train, self.y_train, cv=5,
                                     scoring=["accuracy", "precision", "f1", "recall"])

        y_pred = model.predict(self.X_test)

        # Métricas
        cv_accuracy = np.mean(response_cv["test_accuracy"])
        cv_precision = np.mean(response_cv["test_precision"])
        cv_f1 = np.mean(response_cv["test_f1"])
        cv_recall = np.mean(response_cv["test_recall"])

        test_accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred)
        test_precision = precision_score(y_true=self.y_test, y_pred=y_pred, average="weighted")
        test_f1 = f1_score(y_true=self.y_test, y_pred=y_pred, average="weighted")
        test_recall = recall_score(y_true=self.y_test, y_pred=y_pred, average="weighted")
        test_mcc = matthews_corrcoef(y_true=self.y_test, y_pred=y_pred)
        test_cm = confusion_matrix(self.y_test, y_pred).tolist()

        overfitting_ratio = test_recall / cv_recall
        columns = ["params", "train_accuracy", "train_precision", "train_f1", "train_recall",
                   "test_accuracy", "test_precision", "test_f1", "test_recall", "test_mcc", "test_cm",
                   "overfitting_ratio"]
                    
        row = [[str(best_params), cv_accuracy, cv_precision, cv_f1, cv_recall,
                test_accuracy, test_precision, test_f1, test_recall, test_mcc, test_cm,
                overfitting_ratio]]

        df = pd.DataFrame(columns=columns, data=row)
        df.to_csv(f"{output_path}/results_tuning_{algorithm}.csv", index=False)