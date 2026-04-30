from pyspark.sql import SparkSession  
from pyspark.sql import functions as F  
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler  
from pyspark.sql.types import StringType 

# Initialize Spark session
spark = SparkSession.builder.appName("CategoricalFeatureEngineering").getOrCreate()

# Load dataset (loan_train.csv)
df = spark.read.csv(
    "/opt/spark-apps/loan_train.csv",
    header=True,
    inferSchema=True
)

# Show data
df.show(5)

# Check schema
df.printSchema()

#1.3. Create new dataframe with categorical features
#Select only categorical columns:
categorical_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]  
cdf = df.select(categorical_cols)  
cdf.show(5)

cdf.select(
    [F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c) 
    for c in cdf.columns]
).show()

# Impute missing values
cdf_imputed = cdf.fillna("Unknown")  # Replace nulls with "Unknown"

# Check whether the null values have been imputed
cdf_imputed.select([
    F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c)
    for c in cdf_imputed.columns
]).show()

# Drop Loan_ID
cdf_cleaned = cdf_imputed.drop("Loan_ID")
cdf_cleaned.show(5)

## 2.1. Encode nominal features

nominal_features = ["Gender", "Married", "Self_Employed"]  

# Step 1: Convert strings to numeric indices  
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in nominal_features]  

from pyspark.ml import Pipeline  
pipeline = Pipeline(stages=indexers)  
indexed_df = pipeline.fit(cdf_cleaned).transform(cdf_cleaned)  

# Step 2: One-hot encode indexed values  
encoder = OneHotEncoder(
inputCols=[f"{col}_index" for col in nominal_features],  
outputCols=[f"{col}_encoded" for col in nominal_features]
)  
encoded_nominal_df = encoder.fit(indexed_df).transform(indexed_df)  

nominaldf = encoded_nominal_df.select([f"{col}_encoded" for col in nominal_features])  
nominaldf.show(5)  

## 2.2. Encode ordinal features

from pyspark.sql.types import IntegerType

ordinal_mappings = {  
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},  
    "Education": {"Not Graduate": 0, "Graduate": 1},  
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}  
}  

# Convert mappings to Spark UDFs  
from pyspark.sql.functions import udf  

def map_ordinal(value, mapping):  
    return mapping.get(value, -1)  

map_udf = udf(lambda x, key: map_ordinal(x, ordinal_mappings[key]), returnType=IntegerType())  

# Apply mappings to columns  
ordinaldf = cdf_cleaned  

for col_name in ["Dependents", "Education", "Property_Area"]:  
    ordinaldf = ordinaldf.withColumn(f"{col_name}_encoded", map_udf(F.col(col_name), F.lit(col_name)))  

ordinaldf = ordinaldf.select([f"{col}_encoded" for col in ["Dependents", "Education", "Property_Area"]])  
ordinaldf.show(5)  

## 2.3. Encode target feature

target_indexer = StringIndexer(inputCol="Loan_Status", outputCol="label", handleInvalid="keep")  
target_model = target_indexer.fit(cdf_cleaned)  
target = target_model.transform(cdf_cleaned).select("label")  
target.show(5) 

## 2.4. Combine all encoded categorical features.

final_df = encoded_nominal_df

for col_name in ["Dependents", "Education", "Property_Area"]:
    final_df = final_df.withColumn(
        f"{col_name}_encoded",
        map_udf(F.col(col_name), F.lit(col_name))
    )
final_df = target_model.transform(final_df)

final_columns = (
    [f"{col}_encoded" for col in nominal_features] +
    [f"{col}_encoded" for col in ["Dependents", "Education", "Property_Area"]] +
    ["label"]
)
categoricaldf = final_df.select(*final_columns)
categoricaldf.show(5, truncate=False)