package com.sparkProject

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler,StandardScaler,StringIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator,TrainValidationSplit}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

/**
  * Created by leonardbinet on 02/11/2016.
  */
object JobML {

  def main(args: Array[String]): Unit = {

    // data path should be given as parameter
    if (args.length == 0) {
      println("Usage tp_spark-assembly-1.0.jar file where file is a .parquet file. Example: tp_spark-assembly-1.0.jar /path/to/parquet")
      sys.exit(0)
    }

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .appName("Spark_Star")
      .getOrCreate()

    // Remove INFOs in command line
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    println("\n------------\nREAD FILE: ",args(0),"\n------------------\n")

    // Load clean data
    val df = spark.read.parquet(args(0))
    println("Number of columns: ", df.columns.length)
    println("Number of rows: ", df.count())

    println("\n------------\nPREPROCESSING \n------------------\n")

    // Select feature columns to assemble in a unique vector, the feature vector doesn't need id or label
    val x_colnames_to_scale = df.columns.filter(_ != "koi_disposition").filter(_ != "rowid")

    // Define assembler to compact columns in a single vector
    val assembler = new VectorAssembler()
      .setInputCols(x_colnames_to_scale)
      .setOutputCol("featuresRaw")

    // Apply transformation = add a column "features" on our df => df1.
    val df1 = assembler.transform(df)

    // Create label index 0/1 instead of confirmed/false positive
    val indexer = new StringIndexer()
      .setInputCol("koi_disposition")
      .setOutputCol("label")
      .fit(df1)
    val df2 = indexer.transform(df1)

    // Define scaler to create new scaled column "scaledFeatures"
    val scaler = new StandardScaler()
      .setInputCol("featuresRaw")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false)
      .fit(df2)


    // Normalize each feature to have unit standard deviation.
    val df3 = scaler.transform(df2)

    // We want to predict label with feature information.
    // We keep 10% of the dataset for validation
    // Of the 90% remaining, we'll use 70% of it for training, 30% for crossvalidation to tune hyperparameters

    // We select only features and label columns and make a 90/10 split for training/validation
    val Array(training, validation) = df3.select("features", "label").randomSplit(Array(0.9, 0.1))
    println(training.count())
    println(validation.count())

    println("\n------------\nESTIMATOR, PARAMETERS GRID, AND EVALUATOR DEFINITION \n------------------\n")

    // Define our estimator: Logistic Regression with Lasso regularization
    val lr = new LogisticRegression()
      .setElasticNetParam(1.0) // L1-norm regularization : LASSO
      .setLabelCol("label")
      .setStandardization(true) // to scale each feature of the model
      .setFitIntercept(true) // we want an affine regression (with false, it is a linear regression)
      .setTol(1.0e-5) // stop criterion of the algorithm based on its convergence
      .setMaxIter(300) // a security stop criterion to avoid infinite loops

    // We build a grid of parameters to test to tune Lasso and Ridge parameters
    //val elasticRange = (-6.0 to 0 by 0.5 toArray).map(math.pow(10, _))
    //val regularizationRange = (-6.0 to 0 by 0.5 toArray).map(math.pow(10, _))

    // Short version with fewer parameters
    val elasticRange = (-6.0 to 0 by 3 toArray).map(math.pow(10, _))
    val regularizationRange = (-6.0 to 0 by 3 toArray).map(math.pow(10, _))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, elasticRange)
      .addGrid(lr.regParam,regularizationRange)
      .build()


    // Here: two different methods CrossValidator (with k-folds), of TrainValidationSplit (split once)

    /*
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val crossval = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = crossval.fit(training)

    val resultatCV = cvModel.transform(validation)

    println(final_evaluator.evaluate(resultatCV))

    */
    println("\n------------\nTRAINING \n------------------\n")

    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)

    // Run validation, and choose the best set of parameters.
    val cvModelSplit = trainValidationSplit.fit(training)

    // Finally, we check if our predictions are good on the validation dataset
    // Make predictions on test documents. cvModel uses the best model found.
    val dfWithPrediction = cvModelSplit.transform(validation)

    // Compute score with Evaluator
    println("\n------------\nSCORE \n------------------\n")

    val final_evaluator = new BinaryClassificationEvaluator()
    println(final_evaluator.evaluate(dfWithPrediction))
    dfWithPrediction.groupBy("label", "prediction").count.show()


  }
}
