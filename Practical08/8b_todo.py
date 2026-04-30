from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer, MinMaxScaler
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("AdultFeatureEngineering").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setLogLevel("OFF")

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

for c in df.columns:
    if isinstance(df.schema[c].dataType, StringType):
        df = df.withColumn(c, F.trim(F.col(c)))

df.show(5)
df.printSchema()

# Replace ? with Unknown
df = df.replace("?", "Unknown")

#CATEGORICAL FEATURES

nominal_features = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

#Ordinal feature
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

df = df.withColumn(
    "education_encoded",
    map_education_udf(F.col("education"))
)

# StringIndexer for nominal features
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_index",
        handleInvalid="keep"
    )
    for col in nominal_features
]

pipeline = Pipeline(stages=indexers)
indexed_df = pipeline.fit(df).transform(df)

#OneHotEncoder
encoder = OneHotEncoder(
    inputCols=[f"{col}_index" for col in nominal_features],
    outputCols=[f"{col}_encoded" for col in nominal_features]
)

encoded_df = encoder.fit(indexed_df).transform(indexed_df)

# Encode target
target_indexer = StringIndexer(
    inputCol="income",
    outputCol="label",
    handleInvalid="keep"
)

encoded_df = target_indexer.fit(encoded_df).transform(encoded_df)

#NUMERICAL FEATURES

numerical_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

#Check missing numerical values
encoded_df.select([
    F.count(
        F.when(F.isnan(F.col(c)) | F.col(c).isNull(), c)
    ).alias(c)
    for c in numerical_cols
]).show()

#Impute numerical missing values
imputer = Imputer(
    inputCols=numerical_cols,
    outputCols=[f"{c}_imputed" for c in numerical_cols]
)

imputed_df = imputer.fit(encoded_df).transform(encoded_df)

#Outlier removal using IQR on hours_per_week
numeric_col = "hours_per_week_imputed"

Q1, Q3 = imputed_df.approxQuantile(numeric_col, [0.25, 0.75], 0.05)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

clean_df = imputed_df.filter(
    (F.col(numeric_col) >= lower_limit) &
    (F.col(numeric_col) <= upper_limit)
)

print("Before outlier removal:", imputed_df.count())
print("After outlier removal:", clean_df.count())

#COMBINE ALL FEATURES

feature_cols = (
    [f"{col}_encoded" for col in nominal_features] +
    ["education_encoded"] +
    [f"{col}_imputed" for col in numerical_cols]
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

assembled_df = assembler.transform(clean_df)

#Scale final features
scaler = MinMaxScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

final_model = scaler.fit(assembled_df)
final_df = final_model.transform(assembled_df)

final_df = final_df.select("scaledFeatures", "label")

final_df.show(5, truncate=False)
final_df.printSchema()

print("Final DataFrame is ready for machine learning.")