# %%
import mlflow.sklearn
import pandas as pd
import mlflow

# Import do modelo
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Preprocessor Prod
models = mlflow.search_registered_models(filter_string="name = 'preprocessor_RF'")
latest_version = max([i.version for i in models[0].latest_versions])

preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor_RF/{latest_version}")
preprocessor_feature = preprocessor.feature_names_in_

# Modelo Prod
models = mlflow.search_registered_models(filter_string="name = 'baseline_model_RF'")
latest_version = max([i.version for i in models[0].latest_versions])

model_RF = mlflow.sklearn.load_model(f"models:/baseline_model_RF/{latest_version}")
features = model_RF.feature_names_in_
# %%
test_csv = pd.read_csv("../data/test.csv")
test_csv_no_id = test_csv.drop(columns={"id"})

# 1. Aplicar a transformação (retorna um array)
test_csv_proc_array = preprocessor.transform(test_csv_no_id)

# 2. Converter o Array de volta para DataFrame
# Usamos o get_feature_names_out() do próprio preprocessor para pegar os nomes das colunas processadas
test_csv_proc_df = pd.DataFrame(
    test_csv_proc_array, 
    columns=preprocessor.get_feature_names_out(),
    index=test_csv.index
)

test_csv_final = test_csv_proc_df[features]

# Score
predprob = model_RF.predict_proba(test_csv_final)[:,1]
predict = model_RF.predict(test_csv_final)
# %%
ouput = pd.DataFrame({
    "id":test_csv["id"],
    "target":predprob
})

ouput.to_csv("../data/submission_final.csv", index=False)