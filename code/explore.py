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
data_00 = pd.read_csv("../porto_seguro/data/train.csv")
metadata_00 = generate_metadata(data_00)  # documentation only
ingestion_cols = metadata_00.nome_variavel.to_list()
# %%
prep_cols_to_remove = ["id", "target"]
feature_cols = [x for x in data_00.columns if x not in prep_cols_to_remove]
# %%
x, y = data_00[feature_cols], data_00['target']
# %%
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

# Replace -1 with Null
data_00 = replace_minus_one_for_nan(data_00)
metadata_00 = generate_metadata(data_00)
metadata_00_no_id_tgt = metadata_00.copy().query("nome_variavel != ('id', 'target')")
metadata_00_no_id_tgt
# %%
# Generate High Null % to drop
cutoff_value = 68
cols_high_null_drop = (metadata_00_no_id_tgt[metadata_00_no_id_tgt["percent_nulos"]>= cutoff_value]
                       .nome_variavel
                       .to_list()
                        )
cols_high_null_drop
# %%
# TRATAMENTO DE NULOS
# Fill missing values
# Categorial Replace w MISS_VERIFICAR
# Numerical Replace w MEAN()
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

cat_cols = metadata_00_no_id_tgt[metadata_00_no_id_tgt["tipo"]==np.dtype("object")]
print("qtd cat_cols", len(cat_cols))

if len(cat_cols) > 0:
    cutoff_encoding = 20
    cat_cols_one_hot = cat_cols[cat_cols["cardinalidade"] <= cutoff_encoding]["nome_variavel"].tolist()
    print("qtd cat_cols_one_hot", len(cat_cols_one_hot))

    cat_cols_label_encoding = cat_cols[cat_cols["cardinalidade"] > cutoff_encoding]["nome_variavel"].tolist()
    print("qtd cat_cols_label_encoding", len(cat_cols_label_encoding))
else:
    cat_cols_one_hot = []
    cat_cols_label_encoding = []
    print("No categorical columns found")

num_cols = (metadata_00_no_id_tgt[metadata_00_no_id_tgt["tipo"].isin([np.dtype("float64"), np.dtype("int64")])]
            ["nome_variavel"]
            .to_list()
            )
print("qtd num_cols", len(num_cols))
# %%
# Build the preprocessor
transformers = []

# Numerical pipeline: Impute → Scale
if len(num_cols) > 0:
    transformers.append(
        ("numerical_pipeline", 
         Pipeline([
             ("imputer", SimpleImputer(strategy="mean")),
             ("scaler", StandardScaler())
         ]), 
         num_cols)
    )

# Categorical for OneHot: Impute → OneHotEncode
if len(cat_cols_one_hot) > 0:
    transformers.append(
        ("categorical_one_hot_pipeline",
         Pipeline([
             ("imputer", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR")),
             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
         ]),
         cat_cols_one_hot)
    )

# Categorical for Label Encoding: Impute → OrdinalEncode
if len(cat_cols_label_encoding) > 0:
    transformers.append(
        ("categorical_label_pipeline",
         Pipeline([
             ("imputer", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR")),
             ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
         ]),
         cat_cols_label_encoding)
    )

# Add categorical transformers only if categorical columns exist
if len(cat_cols_one_hot) > 0:
    transformers.append(("categorical_fillna_one_hot_cols", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR"), cat_cols_one_hot))

if len(cat_cols_label_encoding) > 0:
    transformers.append(("categorical_fillna_label_enc_cols", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR"), cat_cols_label_encoding))


preprocessor = ColumnTransformer(transformers=transformers)
preprocessor
# %%
# Fit Transform Treino
scaled_data_train = preprocessor.fit_transform(X_train)
# Nome features saida do preprocessor
transformed_columns_train = preprocessor.get_feature_names_out()
transformed_train = pd.DataFrame(data=scaled_data_train, columns=transformed_columns_train, index=X_train.index)
# %%
#Transform test
scaled_data_test = preprocessor.transform(X_test)
#Features
transformed_columns_test = preprocessor.get_feature_names_out()
#Build Dataframe
transformed_test = pd.DataFrame(data=scaled_data_test, columns=transformed_columns_test, index=X_test.index)
transformed_test
# %%
# Feature Importance no treino em um arvore sem parâmetros
# Para este projeto, 41 features (correspondem a 95% das variáveis explicativas fundamentais para o desenvolvimento do modelo)
from sklearn import tree
model_tree = tree.DecisionTreeClassifier(random_state=42)
model_tree.fit(transformed_train, y_train)

feature_importances = (pd.Series(
    model_tree.feature_importances_, index=transformed_train.columns)
    .sort_values(ascending=False)
    .reset_index()
)

feature_importances["acum"] = feature_importances[0].cumsum()
feature_importances[feature_importances["acum"] < 0.96]

# Best Features
best_features = feature_importances[feature_importances["acum"] < 0.96]["index"].to_list()
# %%
# Tx Variavel Resposta (Aponta para evento raro)
print("Tx Variavel Resposta Geral", y.mean(), "Size", y.shape)
print("Tx Variavel Resposta Treino", y_train.mean(), "Size", y_train.shape)
print("Tx Variavel Resposta Teste", y_test.mean(), "Size", y_test.shape)
# %%
# Treinar o modelo
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

model = LogisticRegression(random_state=42, solver="liblinear")
# model = RandomForestClassifier(min_samples_leaf=40000, n_estimators=1000, random_state=42, criterion="gini", n_jobs=-1)
model.fit(transformed_train[best_features], y_train)
# %%
#Predict e Proba Treino
y_train_predict = model.predict(transformed_train[best_features])
y_train_proba = model.predict_proba(transformed_train[best_features])[:, 1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)
gini_train = 2 * auc_train - 1

print(f"Acurácia treino: {acc_train}")
print(f"AUC treino: {auc_train}")
print(f"Gini Score no Treino: {gini_train:.4f}")
# %%
#Predict e Proba Teste
y_test_predict = model.predict(transformed_test[best_features])
y_test_proba = model.predict_proba(transformed_test[best_features])[:, 1]

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
plt.title(f"Curva ROC - {model.__str__().split('(')[0]}")
plt.legend(
    [
        f"Treino: {100 * auc_train:.2f}",
        f"Teste: {100 * auc_test:.2f}",
        # f"Out-of-Time: {100 * auc_oot:.2f}",
    ]
)
plt.show()
# %%
# Aplicar dados a base de testes e realizar o submission
oot_data_00 = pd.read_csv("../porto_seguro/data/test.csv")

#Transform oot
scaled_data_oot = preprocessor.transform(oot_data_00[feature_cols])

#Features
transformed_columns_oot = preprocessor.get_feature_names_out()

#Build Dataframe
transformed_oot = pd.DataFrame(data=scaled_data_oot, columns=transformed_columns_oot, index=oot_data_00.index)
transformed_oot

# Predicoes
#Predict e Proba OOT
y_oot_predict = model.predict(transformed_oot[best_features])
y_oot_proba = model.predict_proba(transformed_oot[best_features])[:, 1]
submission_df = pd.DataFrame({
    "id": oot_data_00["id"],
    "target": y_oot_proba
})

submission_df.to_csv("../porto_seguro/data/submission_final.csv", index=False)
print("Arquivo de submissão 'submission_final.csv' gerado com sucesso.")