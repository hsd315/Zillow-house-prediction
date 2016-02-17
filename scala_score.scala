

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
import org.apache.spark.ml.feature.{OneHotEncoder, IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.Pipeline

case class QuantileSumamry(
     p5:Double, p10:Double,
     p25:Double,p50:Double,
     p75:Double,p90:Double,
     p95:Double
                          )





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
 // get the datatype for each column
 val dataType = df.schema.map(item=>item.dataType)
 val columnName = df.schema.map(item=>item.name)
 val mapColumn = columnName.zip(dataType)

 val colString = df.schema.filter(item=>item.dataType == StringType).map(item=>item.name)

 val numString = df.schema.filter(item=>item.dataType != StringType).map(item=>item.name)
 //val numStringNew = df.columns.filter(item => item!="target")



 def nullReplace(
                df:DataFrame,
                column:String,
                value:String
                ):Seq = {
  require(df.columns.contains(column),
  "Should provide valid column name")
  val result = df.select(column).na.fill(value.toFloat).asInstanceOf[Seq]
   result

 }

  val newData = numString.map(item => nullReplace(df,item,"1"))


  val dataNew = numString.map(item=>df.select(item).na.fill(1))

  val colsName = Seq( "v1", "v2", "v4")





  def stringColumn(
                 df:DataFrame
                 ):Seq[String]={
   val colString = df.schema.filter(item=>item.dataType == StringType).map(item=>item.name)
   colString

 }

  def numericColumn(
                   df:DataFrame
                   ):Seq[String] ={
    val colNumeric = df.schema.filter(item=>item.dataType !=StringType).map(item=>item.name)
    colNumeric
  }



//// search the N/A values
 val oneColumn = nullReplace(df,"v1",1)


//
 val transformData = quantiled(df,"Field7")//,"Field8"))


 def quantiled(df:DataFrame, column:String):QuantileSumamry = {

  //for (column  <-columns)
   require(df.columns.contains(column), "Should provide valid column name")

   val colTypeMap = Map[String, StructField](df.schema.fields.map(f => (f.name, f)): _*)
   require(colTypeMap(column).dataType.isInstanceOf[NumericType], "Column Type should be NuericType")



   val col = df.select(column).filter(df(column).isNotNull).sort(df.col(column).asc).rdd

   val lookupRdd = col.zipWithIndex().map { case (a: Row, b: Long) => (b, a) }

   val n = lookupRdd.count()

   val p5 = lookupRdd.lookup(n / 20).head.get(0).toString.toDouble
   val p10 = lookupRdd.lookup(n / 10).head.get(0).toString.toDouble
   val p25 = lookupRdd.lookup(n / 4).head.get(0).toString.toDouble
   val p50 = lookupRdd.lookup(n / 2).head.get(0).toString.toDouble
   val p75 = lookupRdd.lookup(3 * n / 4).head.get(0).toString.toDouble
   val p90 = lookupRdd.lookup(9 * n / 10).head.get(0).toString.toDouble

   val p95 = lookupRdd.lookup(19 * n / 20).head.get(0).toString.toDouble


      QuantileSumamry(p5, p10, p25, p50, p75, p90, p95)


 }

//

 def correlation(
                  df: DataFrame,
                  column1: String,
                  column2: String): Double = {
  val colTypeMap = Map[String, StructField](df.schema.fields.map(f => (f.name, f)): _*)

  require(df.columns.contains(column1) && df.columns.contains(column1),
   "Should provide valid column name")

  require(colTypeMap(column1).dataType.isInstanceOf[NumericType] && colTypeMap(column2).
    dataType.isInstanceOf[NumericType], "Column Type should be NumericType")

  df.stat.corr(column1, column2)
 }
//
//
 val result = correlation(df,"v1","v2")


// ///########################
// //################
//
  val testData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(test)
 // val testdata = testData.withColumn("QuoteConversion_Flag",testData("Field7") +1)

  val submitFile = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(submit)

  //########################
  //#####################
  val colsNameTest = Seq( "v1", "v2", "v4")
  //val dataAll = features.select("Original_Quote_Date").cache()
  val testFeatures = testData.select(colsNameTest.head, colsNameTest.tail: _*)


  ///#########
  val columnsName = df.columns


  val submitData = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(submit)
  // drop the two columns
  val cols = df.columns

  val elements = List("target", "ID")

  val target = df.select("target")

  //########################################
  val colsNew = cols.filter(_ != "target").filter(_ != "ID")

  val features = df.select(colsNew.head, colsNew.tail: _*)
  // one hot hash coding to the features

  val colsName = Seq( "v1", "v2", "v4")
  //val dataAll = features.select("Original_Quote_Date").cache()

  val stringIndexers = colsName.map(colName => new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed").fit(df))

  //val indexer = new StringIndexer().setInputCol("Field6").setOutputCol("Field6"+"Index").fit(df.select("Field6"))


  val assemble = new VectorAssembler().setInputCols(Array(colsName: _*)).setOutputCol("Features")


  val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("target" + "Index").fit(df)

  val randomForest = new RandomForestClassifier().setLabelCol("targetIndex").setFeaturesCol("Features")

  val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)


  val pipeline = new Pipeline().setStages(Array.concat(stringIndexers.toArray, Array(labelIndexer, assemble, randomForest, labelConverter)))

  val paramGrid = new ParamGridBuilder().addGrid(randomForest.maxBins, Array(25, 28, 31)).addGrid(randomForest.maxDepth, Array(4, 6, 8)).addGrid(randomForest.impurity, Array("entropy", "gini")).build()


  val evaluator = new BinaryClassificationEvaluator().setLabelCol("targetIndex")

  val evaluator2 = new BinaryClassificationEvaluator().setLabelCol("SurvivedIndexed").setMetricName("areaUnderPR")

  //############################################
  //###########################################
  val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)


  val crossValidatorModel = cv.fit(newData)




//  df.printSchema()
//  val predictions = crossValidatorModel.transform(testdata)
//  predictions.printSchema()
//  val resultFinal = predictions.select("probability")
//
//  val probResult = resultFinal.map(row=>row(0)).map(row=>row.asInstanceOf[DenseVector]).map(row=>row(1)).collect()
////  val model = crossValidatorModel.bestModel
// // predictions.select("probability")
//
//  ///########### submit the file
//  val Id = testData.select("QuoteNumber")
//  val IdArray = Id.map(row=>row(0)).map(row=>row.asInstanceOf[Int]).collect()
//
//  val submit2 = IdArray.zip(probResult)
////  def mergeDataframe(df1:DataFrame,df2:DataFrame):DataFrame = {
//
//  //}
//  val data3 = sc.parallelize(submit2).toDF()
//  data3.printSchema()
//  val data1 = data3.withColumnRenamed("_1","QuoteNumber")
//  val data2 = data1.withColumnRenamed("_2","QuoteConversion_Flag")
//  data2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("/Users/weizhili/Desktop/Spark-Finance/Output2/output.csv")




}
