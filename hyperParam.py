from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_f

from pyspark.sql.functions import abs

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Start Spark
spark = SparkSession.builder.appName("GBT_Tuning").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load data
df = spark.read.option("header", "true").csv("/opt/spark-apps/marks.csv")

# Convert to float
df = df.select([sql_f.col(c).cast("float").alias(c) for c in df.columns])

# Drop unnecessary columns
df = df.drop("Student ID").drop("Attendance (%)")

# Split data
train_df, test_df = df.randomSplit([0.7, 0.3], seed=123)

# Features
feature_cols = train_df.columns
feature_cols.remove("Final Exam Mark")

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# Best model (GBT)
gbt = GBTRegressor(
    featuresCol="scaledFeatures",
    labelCol="Final Exam Mark"
)

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, gbt])

# Hyperparameter grid
paramGrid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5, 7])
    .addGrid(gbt.maxIter, [10, 20, 30])
    .addGrid(gbt.stepSize, [0.05, 0.1])
    .build()
)

# Evaluator
evaluator = RegressionEvaluator(
    labelCol="Final Exam Mark",
    predictionCol="prediction",
    metricName="rmse"
)

# Cross Validation
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# Train
cv_model = cv.fit(train_df)

# Best model
best_model = cv_model.bestModel

# Test evaluation
preds = best_model.transform(test_df)

rmse = evaluator.evaluate(preds)

r2_eval = RegressionEvaluator(
    labelCol="Final Exam Mark",
    predictionCol="prediction",
    metricName="r2"
)
r2 = r2_eval.evaluate(preds)

# Extract best GBT stage
best_gbt = best_model.stages[-1]

# Print ONLY final results
print("\nBEST HYPERPARAMETERS")
print(f"maxDepth = {best_gbt.getMaxDepth()}")
print(f"maxIter  = {best_gbt.getMaxIter()}")
print(f"stepSize = {best_gbt.getStepSize()}")

print("\nMODEL PERFORMANCE")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

spark.stop()