package org.hy.dev.spark

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author edwin
  * @since 18 May 2017
  */
object HashingTF {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("HashingTF")
        conf.setIfMissing("master", "local")
        conf.setIfMissing("spark.master", "local")
        conf.setIfMissing("--num-executors", "3")
        conf.setIfMissing("--driver-memory", "1g")
        conf.setIfMissing("--executor-memory", "1g")
        conf.setIfMissing("--executor-cores", "1")

        val sc = new SparkContext(conf)

        val documents: RDD[Seq[String]] = sc.textFile("/Users/edwin/ln/hy/spark-dev/input/text").map(_.split("\t").toSeq)
        documents.foreach(println)
        val hashingTF = new HashingTF()

        val tf: RDD[Vector] = hashingTF.transform(documents)
        tf.foreach(println)
        val idf = new IDF().fit(tf)

        val tfidf: RDD[Vector] = idf.transform(tf)

        tfidf.foreach(println)

    }
}
