from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors

final_combined_df.write.csv("finaldata.csv", header=True, mode="overwrite")

#Loading data from csv
spark = SparkSession.builder.appName("LogisticRegressionLab").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "/opt/spark-apps/finaldata.csv",
    header=True,
    inferSchema=True
)

df.show(5)
df.printSchema()


df.select([
    F.count(
        F.when(F.col(c).isNull() | F.isnan(F.col(c)), c)
    ).alias(c)
    for c in df.columns
]).show()

feature_cols = [col for col in df.columns if col != "label"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

final_df = assembler.transform(df)
final_df = final_df.select("features", "label")

final_df.show(5)

train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {train_df.count()}, Test records: {test_df.count()}")

lr = LogisticRegression(featuresCol="features", labelCol="label")

lr_model = lr.fit(train_df)

print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

predictions = lr_model.transform(test_df)

predictions.select("features", "label", "prediction").show(5)

evaluator_roc = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",
    labelCol="label",
    metricName="areaUnderROC"
)

print(f"AUC-ROC: {evaluator_roc.evaluate(predictions)}")

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

print(f"Accuracy: {evaluator_acc.evaluate(predictions)}")

confusion_matrix = predictions.groupBy("label").pivot("prediction").count().na.fill(0)
confusion_matrix.show()

new_data = spark.createDataFrame([
    (Vectors.dense([1.0, 0.5, 3000.0, 12000.0, 360, 1, 0]),)
], ["features"])

prediction = lr_model.transform(new_data)

prediction.select("features", "prediction").show()
