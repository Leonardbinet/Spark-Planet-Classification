package com.sparkProject

import org.apache.spark.sql.SparkSession

object Job {

  def main(args: Array[String]): Unit = {

    // SparkSession configuration
    val spark = SparkSession
      .builder
      .getOrCreate()

    // Display only WARNINGS
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._


    // ----------------- Training : word count ------------------------

    val df_wordCount = sc.textFile("/Users/leonardbinet/spark-2.0.0-bin-hadoop2.7/README.md")
      .flatMap{case (line: String) => line.split(" ")}
      .map{case (word: String) => (word, 1)}
      .reduceByKey{case (i: Int, j: Int) => i + j}
      .toDF("word", "count")

    df_wordCount.orderBy($"count".desc).show()


    /********************************************************************************
      *
      *        SPARK Star Classification
      *
      ********************************************************************************/


    // Load CSV in DataFrame
    val df = spark
      .read
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .option("comment", "#") // use # as comment
      .csv("/Users/leonardbinet/Documents/Formation/Cours Telecom/T1_Spark_exo_planete/Data/cumulative.csv")

    // Show number of lines/rows
    //println("display nb of col", df.columns.length)
    //println("nb of rows", df.count)

    // afficher le DF sous forme de table un sous-ensemble des colonnes (exemple les colonnes 10 à 20).
    import org.apache.spark.sql.functions._
    val columns = df.columns.slice(10, 20) // only columns 10 => 20
    df.select(columns.map(col): _*).show(50) // only 50 first rows

    // Afficher le schéma du dataFrame
    df.printSchema()

    // For classification, the number of classes in the training set has to be controled/corrected
    // Lets look at the label classes (colonne koi_disposition).
    df.groupBy($"koi_disposition").count().show()

    // Keep only koi_disposition = CONFIRMED or FALSE POSITIVE
    val df_cleaned = df.filter("koi_disposition in (\"CONFIRMED\", \"FALSE POSITIVE\")")
    df_cleaned.groupBy($"koi_disposition").count().show()

    // Print distinct elements in column “koi_eccen_err1”
    df_cleaned.groupBy($"koi_eccen_err1").count().show()

    // Remove “koi_eccen_err1”.
    val df_cleaned2 = df_cleaned.drop($"koi_eccen_err1")

    // Remove other columns
    val df_cleaned3 = df_cleaned2
      .drop("index")
      .drop("koi_sparprov", "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname", "koi_parm_prov", "koi_limbdark_mod", "koi_fittype", "koi_disp_prov", "koi_comment", "kepoi_name", "kepler_name", "koi_vet_date", "koi_pdisposition")
      .drop("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec")

    println("display nb of col of df_cleaned3 before drop", df_cleaned3.columns.length)

    // Remove useless columns with only one class (=no variance)
    val useless = for(col <- df_cleaned3.columns if df_cleaned3.select(col).distinct().count() == 1 ) yield col
    val df_cleaned4 = df_cleaned3.drop(useless: _*)

    println("display nb of col of df_cleaned3 after drop", df_cleaned4.columns.length)

    df_cleaned4.describe("koi_impact", "koi_duration").show()

    // Replace NA by O
    val df_filled5 = df_cleaned4.na.fill(0.0)

    // Select dataframes like in SQL
    val df_labels = df_filled5.select("rowid", "koi_disposition")
    val df_features = df_filled5.drop("koi_disposition")

    // Join the two last dataframes created. Use Rowid as index
    val df_joined = df_features
      .join(df_labels, usingColumn = "rowid")

    // Functions in Spark are called UDF.
    def udf_sum = udf((col1: Double, col2: Double) => col1 + col2)

    val df_newFeatures = df_filled5
      .withColumn("koi_ror_min", udf_sum($"koi_ror", $"koi_ror_err2"))
      .withColumn("koi_ror_max", $"koi_ror" + $"koi_ror_err1")

    df_newFeatures
      .coalesce(1) // optional : regroup all data in ONE partition, so that results are printed in ONE file
      // >>>> In general this should not be done, only when the data is small enough to fit in the memory of a single machine.
      .write
      .mode("overwrite")
      .option("header", "true")
      .csv("/Users/leonardbinet/Documents/Formation/Cours Telecom/T1_Spark_exo_planete/Data/cleanDF")

  }

}

