
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("NumericalFeatureEngineering").getOrCreate()

df = spark.read.csv("/opt/spark-apps/loan_train.csv", header=True, inferSchema=True)


# Get list of numerical columns
numerical_cols = [field.name for field in df.schema.fields if not isinstance(field.dataType, StringType)]
print("Numerical Columns:", numerical_cols)


# Select only numerical columns
ndf = df.select(numerical_cols)
ndf.show(5)


from pyspark.sql.functions import count, when, isnan


# Count null or NaN values in each numerical column
null_counts = ndf.select([
    count(when(isnan(col(c)) | col(c).isNull(), c)).alias(c) 
    for c in numerical_cols
])
null_counts.show()

from pyspark.ml.feature import Imputer


# Define imputer
imputer = Imputer(
    inputCols=numerical_cols,
    outputCols=[f"{c}_imputed" for c in numerical_cols]
)


# Fit and transform
imputed_ndf = imputer.fit(ndf).transform(ndf)
imputed_ndf.select([f"{c}_imputed" for c in numerical_cols]).show(5)

numeric_col = "ApplicantIncome_imputed"


# Approximate quantiles
quantiles = imputed_ndf.approxQuantile(numeric_col, [0.25, 0.75], 0.05)
Q1, Q3 = quantiles[0], quantiles[1]
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR


print(f"IQR Range for {numeric_col}: Lower = {lower_limit}, Upper = {upper_limit}")


# Filter out rows beyond outlier limits
rdf = imputed_ndf.filter((col(numeric_col) >= lower_limit) & (col(numeric_col) <= upper_limit))


print("Before outlier removal:", imputed_ndf.count())
print("After outlier removal:", rdf.count())


# Find outlier rows
ddf = imputed_ndf.filter((col(numeric_col) < lower_limit) | (col(numeric_col) > upper_limit))
ddf.select(numeric_col).show()


import matplotlib.pyplot as plt


# Convert small sample to pandas for visualization
rdf_pd = rdf.select(numeric_col).toPandas()


plt.boxplot(rdf_pd[numeric_col])
plt.title("ApplicantIncome After Outlier Removal")
plt.show()


# Extract IDs of removed rows (if ID column exists)
# ddf_ids = ddf.select("id")  # Replace "id" with actual identifier


# Drop those rows from categorical DataFrame
# clean_categoricaldf = clean_categoricaldf.join(ddf_ids, on="id", how="left_anti")


numeric_col = "LoanAmount_imputed"
rdf.select(numeric_col).summary("count", "min", "max", "mean", "stddev").show()

from pyspark.sql.functions import log, sqrt


log_df = rdf.withColumn(f"{numeric_col}_log", log(col(numeric_col)))
log_df.select(f"{numeric_col}_log").show(5)

reciprocal_df = rdf.withColumn(f"{numeric_col}_reciprocal", 1 / col(numeric_col))
reciprocal_df.select(f"{numeric_col}_reciprocal").show(5)

sqrt_df = rdf.withColumn(f"{numeric_col}_sqrt", sqrt(col(numeric_col)))
sqrt_df.select(f"{numeric_col}_sqrt").show(5)

cbrt_df = rdf.withColumn(f"{numeric_col}_cbrt", col(numeric_col) ** (1/3))
cbrt_df.select(f"{numeric_col}_cbrt").show(5)

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler


# Assemble features
assembler = VectorAssembler(inputCols=[f"{c}_imputed" for c in numerical_cols], outputCol="features")
scaled_df = assembler.transform(rdf)


# Normalize
minmax_scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
mm_model = minmax_scaler.fit(scaled_df)
mm_scaled_df = mm_model.transform(scaled_df).select("scaledFeatures")
mm_scaled_df.show(5)


from pyspark.ml.feature import StandardScaler


standard_scaler = StandardScaler(inputCol="features", outputCol="standardizedFeatures")
std_model = standard_scaler.fit(scaled_df)
std_scaled_df = std_model.transform(scaled_df).select("standardizedFeatures")
std_scaled_df.show(5)


# Manual robust scaling
median = rdf.approxQuantile(numeric_col, [0.5], 0.01)[0]
q1 = rdf.approxQuantile(numeric_col, [0.25], 0.01)[0]
q3 = rdf.approxQuantile(numeric_col, [0.75], 0.01)[0]
iqr = q3 - q1


robust_df = rdf.withColumn(f"{numeric_col}_robust", (col(numeric_col) - median) / iqr)
robust_df.select(f"{numeric_col}_robust").show(5)


#Combine with Categorical Features
from pyspark.sql import DataFrame


# Assume both dfs share an 'id' column
# final_numericaldf = mm_scaled_df.withColumn("id", F.monotonically_increasing_id())  # If no ID exists
# final_combined_df = final_numericaldf.join(clean_categoricaldf, on="id")
# final_combined_df.show(5)

