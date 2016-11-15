#Spark Classification for planets.

 Classification of exo-planets using Spark to implement logistic regression with elastic net penalisation.

## Usage
To create jars: (from build.sbt folder)
```
sbt assembly
```

To launch Job, from spark-submit folder (spark bin):

For first Job:
```
./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 3G --executor-memory 4G --class com.sparkProject.Job /Users/leonardbinet/Documents/Formation/Cours\ Telecom/T1_Spark_exo_planete/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar
```
For machine learning JobML:
```
./spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir="/tmp" --driver-memory 3G --executor-memory 4G --class com.sparkProject.JobML /Users/leonardbinet/Documents/Formation/Cours\ Telecom/T1_Spark_exo_planete/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar /Users/leonardbinet/Documents/Formation/Cours\ Telecom/T1_Spark_exo_planete/Data/cleanedDataFrame.parquet
```

Where:
```/Users/leonardbinet/Documents/Formation/Cours\ Telecom/T1_Spark_exo_planete/tp_spark/target/scala-2.11/tp_spark-assembly-1.0.jar ```
needs to be replaced by your own JAR.

And
```/Users/leonardbinet/Documents/Formation/Cours\ Telecom/T1_Spark_exo_planete/Data/cleanedDataFrame.parquet```
needs to be replace by your own parquet file.

## Environment

 - IntelliJ / or Databricks for testing
 - Spark : ML library
 - Scala language


## Results

```
------------
SCORE
------------------

0.9872352138359732
+-----+----------+-----+
|label|prediction|count|
+-----+----------+-----+
|  1.0|       1.0|  216|
|  0.0|       1.0|   24|
|  1.0|       0.0|    7|
|  0.0|       0.0|  367|
+-----+----------+-----+

```

## Todo
