# ==============================
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# TODO: Classification on Iris Dataset
# ==============================

from sklearn.datasets import load_iris
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load Iris Dataset
iris_data = load_iris()

iris_pd = pd.DataFrame(
    iris_data.data,
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

iris_pd["label"] = iris_data.target

# Convert Pandas DataFrame to Spark DataFrame
iris_df = spark.createDataFrame(iris_pd)

iris_df.show(5)
iris_df.printSchema()

# Combine feature columns into one vector
feature_cols = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

iris_final_df = assembler.transform(iris_df).select("features", "label")

iris_final_df.show(5)

# Split into train and test set
iris_train_df, iris_test_df = iris_final_df.randomSplit([0.8, 0.2], seed=42)

print(f"Iris Training records: {iris_train_df.count()}")
print(f"Iris Test records: {iris_test_df.count()}")

# Train Logistic Regression model
iris_lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    family="multinomial"
)

iris_model = iris_lr.fit(iris_train_df)

# Predict on train set
iris_train_predictions = iris_model.transform(iris_train_df)

# Predict on test set
iris_test_predictions = iris_model.transform(iris_test_df)

iris_test_predictions.select("features", "label", "prediction").show(10)

# Evaluate accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

train_accuracy = accuracy_evaluator.evaluate(iris_train_predictions)
test_accuracy = accuracy_evaluator.evaluate(iris_test_predictions)

print(f"Iris Train Accuracy: {train_accuracy}")
print(f"Iris Test Accuracy: {test_accuracy}")