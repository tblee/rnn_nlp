package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/12/16.
  */

import scala.math
import java.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, BlockMatrix}


object shallowNN_mllib {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("shallowNN_mllib")
    val spark = new SparkContext(conf)

    val x = spark.textFile("spiral_x.txt").map(row => row.split('\t').map(_.toDouble))
    val y = spark.textFile("spiral_y.txt").map(row => row.split('\t').map(_.toDouble))

    // convert raw data into RDD of indexed rows
    val x_idrow = x.zipWithIndex.map{
      case (row, id) => new IndexedRow(id, Vectors.dense(row))
    }
    val y_idrow = y.zipWithIndex.map{
      case (row, id) => new IndexedRow(id, Vectors.dense(row))
    }

    // convert RDD of indexed rows to block matrices
    val x_mat = (new IndexedRowMatrix(x_idrow)).toBlockMatrix()
    val y_mat = (new IndexedRowMatrix(y_idrow)).toBlockMatrix()

    x_mat.validate()
    y_mat.validate()

  }


}
