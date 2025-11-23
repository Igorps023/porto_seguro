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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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

# Build the preprocessor
transformers = [
    ("numerical_fillna_cols", SimpleImputer(strategy="mean"), num_cols),
]

# Add categorical transformers only if categorical columns exist
if len(cat_cols_one_hot) > 0:
    transformers.append(("categorical_fillna_one_hot_cols", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR"), cat_cols_one_hot))

if len(cat_cols_label_encoding) > 0:
    transformers.append(("categorical_fillna_label_enc_cols", SimpleImputer(strategy="constant", fill_value="MISS_VERIFICAR"), cat_cols_label_encoding))


preprocessor = ColumnTransformer(transformers=transformers)
preprocessor
# %%
# Aplicando StandardScaler
scaled_data = preprocessor.fit_transform(X_train[num_cols])
pd.DataFrame(data=scaled_data, columns=X_train.columns, index=X_train.index)
# %%
# PADRONIZACAO
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler
data_02 = scaler.fit_transform(data_01)
data_02
# %%
# TRANSFORM X_train
# X_processed = preprocessor.fit_transform(X_train)
# X_processed
# preprocessor.get_feature_names_out()

# %%
###################
# FEATURE IMPORTANCE

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

feature_importances = (
    pd.Series(arvore.feature_importances_, index=X_train.columns)
    .sort_values(ascending=False)
    .reset_index()
)

feature_importances["acum"] = feature_importances[0].cumsum()
feature_importances[feature_importances["acum"] < 0.96]
# %%
# Tx Variavel Resposta
print("Tx Variavel Resposta Geral", y.mean(), "Size", y.shape)
print("Tx Variavel Resposta Treino", y_train.mean(), "Size", y_train.shape)
print("Tx Variavel Resposta Teste", y_test.mean(), "Size", y_test.shape)
# %%
# DataPrep


filtered_df = replace_minus_one_for_nan(filtered_df)
filtered_df
# %%
# replace -1 with null
data_01 = data_00.copy()
# data_01 = df_train_01.replace(-1, np.nan, inplace=True)
# %%
data_02 = data_00.drop(columns=["id", "target"], axis=1)
# %%
# %%
metadata = generate_metadata(data_00)
metadata
