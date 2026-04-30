from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("AdultCategoricalFeatureEngineering").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setLogLevel("OFF")


# 1. Load Adult dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = spark.read.csv(
    "/opt/adult/adult.data",
    header=False,
    inferSchema=True
).toDF(*columns)

# remove extra spaces from string columns
for c in df.columns:
    if isinstance(df.schema[c].dataType, StringType):
        df = df.withColumn(c, F.trim(F.col(c)))

df.show(5)
df.printSchema()

# 2. Select categorical columns
categorical_cols = [
    field.name for field in df.schema.fields
    if isinstance(field.dataType, StringType)
]

cdf = df.select(categorical_cols)
cdf.show(5)

# 3. Check missing values
# Adult dataset uses ? as missing value
cdf.select([
    F.count(
        F.when((F.col(c).isNull()) | (F.col(c) == "?"), c)
    ).alias(c)
    for c in cdf.columns
]).show()

# 4. Impute missing categorical values
cdf_imputed = cdf.replace("?", "Unknown")

cdf_imputed.select([
    F.count(
        F.when((F.col(c).isNull()) | (F.col(c) == "?"), c)
    ).alias(c)
    for c in cdf_imputed.columns
]).show()

# 5. Separate target column
target_col = "income"

# Nominal categorical features
nominal_features = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

# Ordinal feature
ordinal_features = ["education"]

# 6. Encode nominal features using StringIndexer + OneHotEncoder
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_index",
        handleInvalid="keep"
    )
    for col in nominal_features
]

pipeline = Pipeline(stages=indexers)
indexed_df = pipeline.fit(cdf_imputed).transform(cdf_imputed)

encoder = OneHotEncoder(
    inputCols=[f"{col}_index" for col in nominal_features],
    outputCols=[f"{col}_encoded" for col in nominal_features]
)

encoded_nominal_df = encoder.fit(indexed_df).transform(indexed_df)

# 7. Encode ordinal feature manually
education_mapping = {
    "Preschool": 0,
    "1st-4th": 1,
    "5th-6th": 2,
    "7th-8th": 3,
    "9th": 4,
    "10th": 5,
    "11th": 6,
    "12th": 7,
    "HS-grad": 8,
    "Some-college": 9,
    "Assoc-voc": 10,
    "Assoc-acdm": 11,
    "Bachelors": 12,
    "Masters": 13,
    "Prof-school": 14,
    "Doctorate": 15
}

def map_education(value):
    return education_mapping.get(value, -1)

map_education_udf = F.udf(map_education, IntegerType())

encoded_df = encoded_nominal_df.withColumn(
    "education_encoded",
    map_education_udf(F.col("education"))
)

# 8. Encode target income column
target_indexer = StringIndexer(
    inputCol="income",
    outputCol="label",
    handleInvalid="keep"
)

final_cat_df = target_indexer.fit(encoded_df).transform(encoded_df)

# 9. Combine encoded categorical features into one vector
feature_cols = [f"{col}_encoded" for col in nominal_features] + ["education_encoded"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="categorical_features"
)

final_df = assembler.transform(final_cat_df)

final_df = final_df.select("categorical_features", "label")

final_df.show(5, truncate=False)
final_df.printSchema()