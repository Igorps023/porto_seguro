# %%
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# %%
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from code_functions.prd_functions import generate_metadata


def replace_minus_one_for_nan(dataframe: pd.DataFrame):
    dataframe_01 = dataframe.copy()
    dataframe_01.replace(-1, np.nan, inplace=True)
    return dataframe_01

# %%
data_00 = pd.read_csv("../data/train.csv")
data_00 = replace_minus_one_for_nan(data_00)
# %%
prep_cols_to_remove = ["id", "target"]
feature_cols = [x for x in data_00.columns if x not in prep_cols_to_remove]
# %%
x, y = data_00[feature_cols], data_00['target']

# %% [5] Divisão em Treino e Teste (Split)
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

# %%
metadata_train = generate_metadata(pd.concat([X_train, y_train], axis=1))
metadata_train = metadata_train.query("nome_variavel not in @prep_cols_to_remove").copy()

# %% [6] Identificação de Colunas com Alto Índice de Nulos
# Usamos o cutoff de 68% para listar colunas que serão descartadas
cutoff_value = 68
cols_high_null_drop = (
    metadata_train[metadata_train["percent_nulos"] >= cutoff_value]
    .nome_variavel
    .to_list()
)

print(f"Colunas a serem removidas (> {cutoff_value}% nulos): {cols_high_null_drop}")

# %% [7] Filtragem de Colunas e Definição de Tipos
# 1. Identificar as colunas que sobrevivem ao cutoff (as que vamos manter)
cols_to_keep = [col for col in feature_cols if col not in cols_high_null_drop]

# 2. Filtrar os metadados apenas para as colunas que ficaram
metadata_filtered = metadata_train[metadata_train["nome_variavel"].isin(cols_to_keep)]

# 3. Identificar Categóricas e Numéricas (Baseado nos metadados filtrados)
cat_cols_df = metadata_filtered[metadata_filtered["tipo"] == np.dtype("object")]

if len(cat_cols_df) > 0:
    cutoff_encoding = 20
    cat_cols_one_hot = cat_cols_df[cat_cols_df["cardinalidade"] <= cutoff_encoding]["nome_variavel"].tolist()
    cat_cols_label_encoding = cat_cols_df[cat_cols_df["cardinalidade"] > cutoff_encoding]["nome_variavel"].tolist()
else:
    cat_cols_one_hot = []
    cat_cols_label_encoding = []

# 4. Identificar Numéricas
num_cols = (
    metadata_filtered[metadata_filtered["tipo"].isin([np.dtype("float64"), np.dtype("int64")])]
    ["nome_variavel"]
    .to_list()
)

# Cols to pass for fit
cols_to_process = num_cols + cat_cols_one_hot + cat_cols_label_encoding

# Logs de conferência
print(f"Total de colunas removidas: {len(cols_high_null_drop)}")
print(f"Total de colunas mantidas: {len(cols_to_keep)}")
print(f"---")
print(f"Colunas numéricas: {len(num_cols)}")
print(f"Colunas categóricas (OneHot): {len(cat_cols_one_hot)}")
print(f"Colunas categóricas (LabelEncoding): {len(cat_cols_label_encoding)}")

# %%
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

transformers = []

# Pipeline Numérico: Imputação pela Média + Padronização
if len(num_cols) > 0:
    transformers.append(
        ("numerical_pipeline", 
         Pipeline([
             ("imputer", SimpleImputer(strategy="mean")),
             ("scaler", StandardScaler())
         ]), 
         num_cols)
    )

# Pipeline Categórico OneHot: Imputação com constante + Encoding
if len(cat_cols_one_hot) > 0:
    transformers.append(
        ("categorical_one_hot_pipeline",
         Pipeline([
             ("imputer", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR")),
             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
         ]),
         cat_cols_one_hot)
    )

# Pipeline Categórico Label: Imputação com constante + Ordinal Encoding
if len(cat_cols_label_encoding) > 0:
    transformers.append(
        ("categorical_label_pipeline",
         Pipeline([
             ("imputer", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR")),
             ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
         ]),
         cat_cols_label_encoding)
    )

# Inicialização do ColumnTransformer
preprocessor = ColumnTransformer(transformers=transformers)
preprocessor

# %%
scaled_train = preprocessor.fit_transform(X_train[cols_to_process])

preprocessor_features = preprocessor.get_feature_names_out()

scaled_test = preprocessor.transform(X_test[cols_to_process])

X_train_proc = pd.DataFrame(
    data=scaled_train, 
    columns=preprocessor_features, 
    index=X_train.index
    )

X_test_proc = pd.DataFrame(
    data=scaled_test, 
    columns=preprocessor_features, 
    index=X_test.index
    )

# %%
# Feature Importance via Decision Tree
from sklearn import tree

# Treinamos a árvore para identificar relevância
model_tree = tree.DecisionTreeClassifier(random_state=42)
model_tree.fit(X_train_proc, y_train)

# Criamos a tabela de importância
feature_importances = (
    pd.Series(model_tree.feature_importances_, index=X_train_proc.columns)
    .sort_values(ascending=False)
    .reset_index()
)
feature_importances.columns = ['feature', 'importance']

# Calculamos o acumulado
feature_importances["acum"] = feature_importances['importance'].cumsum()

# Definimos as Best Features (96% da variância explicativa)
# Usamos .iloc para garantir que pegamos até o ponto onde o acumulado atinge o corte
best_features = feature_importances[feature_importances["acum"] < 0.96]["feature"].to_list()

print(f"Total de features após processamento: {len(preprocessor_features)}")
print(f"Features selecionadas (best_features): {len(best_features)}")

# %%
# Tx Variavel Resposta (Aponta para evento raro)
print("Tx Variavel Resposta Geral", y.mean(), "Size", y.shape)
print("Tx Variavel Resposta Treino", y_train.mean(), "Size", y_train.shape)
print("Tx Variavel Resposta Teste", y_test.mean(), "Size", y_test.shape)
# %%
# RandomSearch
import time
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

base_pipe = Pipeline([
    ('clf', LogisticRegression()),
])

param_dist_random = [
    {
        'clf': [LogisticRegression(solver='liblinear', random_state=42)],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], # Range
        'clf__penalty': ['l1', 'l2']
    }
]

# %% [Random Search] - Execução
print("Iniciando Randomized Search...")
start_time = time.time()

random_search = RandomizedSearchCV(
    base_pipe, 
    param_distributions=param_dist_random, 
    n_iter=15, # Testará 15 combinações aleatórias
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=2,
    random_state=42
)

random_search.fit(X_train_proc[best_features], y_train)
random_time = time.time() - start_time

print(f"Random Search finalizado em: {random_time:.2f} segundos")
print(f"Melhor Score Random: {random_search.best_score_:.4f}")
print(f"Melhores Parâmetros Random: {random_search.best_params_}")

# %%
# GRID
print("Iniciando Grid Search Refinado...")
start_time_grid = time.time()

param_grid_refined = [
    {
        'clf': [LogisticRegression(solver='liblinear', random_state=42)],
        'clf__C': [0.005, 0.075, 0.01, 0.02, 0.05], # Range amplo
        'clf__penalty': ['l1']
    }
]

# Execução da Busca (Grid Search)
# scoring='roc_auc' pois é a métrica alvo do dataset
grid_search = GridSearchCV(
    base_pipe, 
    param_grid_refined, 
    cv=5, 
    scoring='roc_auc', 
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train_proc[best_features], y_train)
grid_time = time.time() - start_time_grid

print(f"\nGrid Search finalizado em: {grid_time:.2f} segundos")
print(f"Melhor AUC Final: {grid_search.best_score_:.4f}")
print(f"Melhores Parâmetros Finais: {grid_search.best_params_}")

# %% [12] Avaliação do Melhor Modelo Encontrado
model_final = grid_search.best_estimator_

print(f"Melhor Modelo: {grid_search.best_params_}")
print(f"Melhor AUC (Validação Cruzada): {grid_search.best_score_:.4f}")

# %%
#Predict e Proba Treino
y_train_predict = model_final.predict(X_train_proc[best_features])
y_train_proba = model_final.predict_proba(X_train_proc[best_features])[:, 1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)
gini_train = 2 * auc_train - 1

print(f"Acurácia treino: {acc_train}")
print(f"AUC treino: {auc_train}")
print(f"Gini Score no Treino: {gini_train:.4f}")
# %%
#Predict e Proba Teste
y_test_predict = model_final.predict(X_test_proc[best_features])
y_test_proba = model_final.predict_proba(X_test_proc[best_features])[:, 1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
roc_test = metrics.roc_curve(y_test, y_test_proba)
gini_test = 2 * auc_test - 1

print(f"Acurácia Teste: {acc_test}")
print(f"AUC Teste: {auc_test}")
print(f"Gini Score no Teste: {gini_test:.4f}")
# %%
# --- AVALIAÇÃO DE OVERFITTING ---
print(f"\nDiferença (Treino - Teste): {auc_train - auc_test:.4f}")
# %%
import matplotlib.pyplot as plt

# PLOTAR GRAFICOS
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
# plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0, 1], [0, 1], "--", color="black")
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title(f"Curva ROC - {model_final.__str__().split('(')[0]}")
plt.legend(
    [
        f"Treino: {100 * auc_train:.2f}",
        f"Teste: {100 * auc_test:.2f}",
        # f"Out-of-Time: {100 * auc_oot:.2f}",
    ]
)
plt.show()