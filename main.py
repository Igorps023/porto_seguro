# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn import tree
# %%

spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "1g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
# %%
# %%
dados = (spark.read
         .format("csv")
         .option("inferSchema", "false")
         .load("data/train.csv")
         )
# %%
amostra = dados.limit(10)
# %%
amostra_df = amostra.toPandas()
# %%
tree.DecisionTreeClassifier()
