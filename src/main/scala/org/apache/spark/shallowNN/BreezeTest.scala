package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/13/16.
  */

import scala.math
import scala.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD
import breeze.linalg._


object BreezeTest {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("shallowNN_mllib")
    val spark = new SparkContext(conf)

    // create a scala random number generator
    val rg = new Random()

    val m1 = DenseMatrix.zeros[Double](2, 5).map(_ => rg.nextGaussian())
    println(m1.toString())

    // test create an RDD[DenseMatrix]
    val x = spark.textFile("spiral_x.txt").map(row => DenseMatrix(row.split('\t').map(_.toDouble)))
    val y = spark.textFile("spiral_y.txt").map(row => DenseMatrix(row.split('\t').map(_.toDouble)))

    val m2 = x.map(row => row * m1)
    println(m2)

  }
}
