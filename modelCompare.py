from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_f

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark (suppress logs)
spark = SparkSession.builder.appName("ModelComparison").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load data
df = spark.read.option("header", "true").csv("/opt/spark-apps/marks.csv")

# Convert to float
df = df.select([sql_f.col(c).cast("float").alias(c) for c in df.columns])

# Drop unwanted columns
df = df.drop("Student ID").drop("Attendance (%)")

# Split data
train_df, test_df = df.randomSplit([0.7, 0.3], seed=123)

# Features
feature_cols = train_df.columns
feature_cols.remove("Final Exam Mark")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Evaluators
rmse_eval = RegressionEvaluator(
    labelCol="Final Exam Mark", predictionCol="prediction", metricName="rmse"
)

r2_eval = RegressionEvaluator(
    labelCol="Final Exam Mark", predictionCol="prediction", metricName="r2"
)

# Models
models = {
    "Linear Regression": LinearRegression(featuresCol="scaledFeatures", labelCol="Final Exam Mark"),
    "Decision Tree": DecisionTreeRegressor(featuresCol="scaledFeatures", labelCol="Final Exam Mark"),
    "Random Forest": RandomForestRegressor(featuresCol="scaledFeatures", labelCol="Final Exam Mark"),
    "GBT": GBTRegressor(featuresCol="scaledFeatures", labelCol="Final Exam Mark")
}

results = []

# Train + Evaluate
for name, model in models.items():
    pipeline = Pipeline(stages=[assembler, scaler, model])
    trained = pipeline.fit(train_df)
    preds = trained.transform(test_df)

    rmse = rmse_eval.evaluate(preds)
    r2 = r2_eval.evaluate(preds)

    results.append((name, rmse, r2))

# Print ONLY final output
print("\nMODEL RESULTS")
for name, rmse, r2 in results:
    print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")

best = min(results, key=lambda x: x[1])

print("\nBEST MODEL")
print(f"{best[0]} (RMSE={best[1]:.4f}, R2={best[2]:.4f})")

spark.stop()