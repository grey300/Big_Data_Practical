from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName("LinearRegression").getOrCreate()
df=spark.read.format("csv").option("header","true").load("/opt/spark-apps/marks.csv")

rmse_evaluator = RegressionEvaluator(
    labelCol="Final Exam Mark",
    predictionCol = "prediction",
    metricName = "rmse"
)

df = df.select([sql_f.col(c).cast("float").alias(c) for c in df.columns])
df.printSchema()

df.columns
df = df.drop("Student ID").drop("Attendance (%)")

train_df, test_df = df.randomSplit([0.7,0.3], seed=123456)

feature_cols = train_df.columns
feature_cols.remove('Final Exam Mark')
feature_cols

assembler = VectorAssembler(
    inputCols = feature_cols,
    outputCol = "features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=True
)

assembler.transform(train_df).show()

lr = LinearRegression(
    maxIter=10,
    regParam=0.3,
    featuresCol='scaledFeatures',
    labelCol='Final Exam Mark'
)

# lr_model = lr.fit(assembler.transform(train_df))

# predictions = lr_model.transform(assembler.transform(test_df))
# predictions.select("prediction")

pipeline = Pipeline(stages=[assembler,scaler,lr])
pipeline_model = pipeline.fit(train_df)

prediction = pipeline_model.transform(test_df)
prediction.select('prediction', 'Final Exam Mark').show(5)

pipeline_model.stages[2]
print(pipeline_model.stages[2].coefficients)
print(pipeline_model.stages[2].intercept)

rmse = rmse_evaluator.evaluate(prediction)
print(f"Root Mean Squared Error: {rmse}")


df.cache() #Stores temporily in memory