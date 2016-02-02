

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.{min,max,lit}
import org.apache.spark.ml.classification.RandomForestClassifier

import org.apache.spark.sql.{DataFrame,Column}
import org.apache.log4j.{Logger, Level}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.Pipeline

/**
  * Created by weizhili on 1/19/16.
  *
  *
  *
  */



object dataLoad {
  //  An existing SparkContext.
  val sparkConf = new SparkConf().setAppName("caseDemo").setMaster("local[4]")
  // run four threads (Clusters)
  val sc = new SparkContext(sparkConf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._


  val suning = "/Users/weizhili/Desktop/data/fraud.csv"
  val data = sc.textFile(suning)
  val training = "/Users/weizhili/Desktop/Spark-Finance/train.csv"
  val submit = "/Users/weizhili/Desktop/Spark-Finance/sample_submission.csv"
  val test = "/Users/weizhili/Desktop/Spark-Finance/test.csv"

 //###########
  val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(training)
  df.printSchema()

  val testData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(test)
  val testdata = testData.withColumn("QuoteConversion_Flag",testData("Field7") +1)

  val submitFile = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(submit)

  //########################
  //#####################
  val colsNameTest = Seq( "Field7", "Field8", "Field9")
  //val dataAll = features.select("Original_Quote_Date").cache()
  val testFeatures = testData.select(colsNameTest.head, colsNameTest.tail: _*)


  ///#########
  val columnsName = df.columns


  val submitData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(submit)
  // drop the two columns
  val cols = df.columns

  val elements = List("QuoteConversion_Flag", "QuoteNumber")

  val target = df.select("QuoteConversion_Flag")


  //########################################
  val colsNew = cols.filter(_ != "QuoteConversion_Flag").filter(_ != "QuoteNumber")

 // val features = df.select(colsNew.head, colsNew.tail: _*)
  // one hot hash coding to the features

  val categoricalFeatColNames = features.columns
  val colsName = Seq( "Field7", "Field8", "Field9")
  //val dataAll = features.select("Original_Quote_Date").cache()
  val features = df.select(colsName.head, colsName.tail: _*)

  val stringIndexers = colsName.map(colName => new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed").fit(df))

  //val indexer = new StringIndexer().setInputCol("Field6").setOutputCol("Field6"+"Index").fit(df.select("Field6"))


  val assemble = new VectorAssembler().setInputCols(Array(colsName: _*)).setOutputCol("Features")


  val labelIndexer = new StringIndexer().setInputCol("QuoteConversion_Flag").setOutputCol("QuoteConversion_Flag" + "Index").fit(df)

  val randomForest = new RandomForestClassifier().setLabelCol("QuoteConversion_FlagIndex").setFeaturesCol("Features")

  val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


  val pipeline = new Pipeline().setStages(Array.concat(stringIndexers.toArray, Array(labelIndexer, assemble, randomForest, labelConverter)))

  val paramGrid = new ParamGridBuilder().addGrid(randomForest.maxBins, Array(25, 28, 31)).addGrid(randomForest.maxDepth, Array(4, 6, 8)).addGrid(randomForest.impurity, Array("entropy", "gini")).build()


  val evaluator = new BinaryClassificationEvaluator().setLabelCol("QuoteConversion_FlagIndex")

  val evaluator2 = new BinaryClassificationEvaluator().setLabelCol("SurvivedIndexed").setMetricName("areaUnderPR")

  //############################################
  //###########################################
  val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
  val crossValidatorModel = cv.fit(df)
  df.printSchema()
  val predictions = crossValidatorModel.transform(testdata)
  predictions.printSchema()
  val result = predictions.select("probability")

  val probResult = result.map(row=>row(0)).map(row=>row.asInstanceOf[DenseVector]).map(row=>row(1)).collect()
//  val model = crossValidatorModel.bestModel
 // predictions.select("probability")

  ///########### submit the file
  val Id = testData.select("QuoteNumber")
  val IdArray = Id.map(row=>row(0)).map(row=>row.asInstanceOf[Int]).collect()

  val submit = IdArray.zip(probResult)
//  def mergeDataframe(df1:DataFrame,df2:DataFrame):DataFrame = {

  //}
  val data = sc.parallelize(submit).toDF()
  data.printSchema()
  val data1 = data.withColumnRenamed("_1","QuoteNumber")
  val data2 = data1.withColumnRenamed("_2","QuoteConversion_Flag")
  data2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("/Users/weizhili/Desktop/Spark-Finance/Output2/output.csv")

  //case class rowPredict(QuoteNumber:Int,QuoteConversion_Flag:Double)
 // val ds = submit.map(row=>Seq(rowPredict(row._1,row._2)))



}
