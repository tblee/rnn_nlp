package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/10/16.
  */

import scala.math
import scala.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD

object MatrixTest {

  class Matrix(val m: RDD[Array[Double]],
               val nrow: Int, val ncol: Int) {
    // Class Matrix
    // takes an RDD as input and the constructor also requires the shape
    // of matrix explicitly specified
    // --> re-calculating the shape is expensive

    def shape: (Int, Int) = (nrow, ncol)


    def transpose: Matrix = {
      val indexed = m.zipWithIndex.flatMap{
        case (row, rowID) => row.zipWithIndex.map{
          case (num, colID) => colID -> (rowID, num)
        }
      } // pair each matrix element with row and column ID

      val byCol = indexed.groupByKey.sortByKey().values
      val transposed = byCol.map(row => row.toArray.sortBy(_._1).map(_._2))
      new Matrix(transposed, ncol, nrow)
    }

    // matrix addition
    def +(other: Matrix): Matrix = {

      if (ncol == other.ncol) {
        if (other.nrow == 1) {
          // if ncol matches and the target matrix is a single row matrix
          // repeat the target rows to match dimensions
          val otherRow = other.m.collect.flatten
          val added = m.map(row => row.zip(otherRow).map(pair => pair._1 + pair._2))
          new Matrix(added, nrow, ncol)

        } else
          if (nrow == other.nrow) {

            val sum = m.zip(other.m).map{
              case (arr1, arr2) => arr1.zip(arr2).map {
                case (num1, num2) => num1 + num2
              }
            }
            new Matrix(sum, nrow, ncol)
          }
          else
            throw new IllegalArgumentException("Dimensions do not match")
        } else
          throw new IllegalArgumentException("Dimensions do not match!")
    }

    // matrix addition for Array[Array[Double]] operand
    def +(other: Array[Array[Double]]): Matrix = {

      if (ncol == other.head.size) {
        if (other.size == 1) {
            val o = other.head
            val sum = m.map(row => row.zip(o).map(pair => pair._1 + pair._2))
            new Matrix(sum, nrow, ncol)
        } else
          if (nrow == other.size) {
            val sum = m.zipWithIndex.map{
              case (row, index) => row.zip(other(index.toInt))
            }.map(row => row.map(pair => pair._1 + pair._2))
            new Matrix(sum, nrow, ncol)
        } else
          throw new IllegalArgumentException("Dimensions do not match!")
      } else
        throw new IllegalArgumentException("Dimensions do not match!")
    }

    // scalar multiplication
    def *(s: Double): Matrix = {
      val res = m.map(row => row.map(_ * s))
      new Matrix(res, nrow, ncol)
    }

    // element-wise multiplication
    def *(other: Matrix): Matrix = {
      if (nrow == other.nrow) {

        val res = m.zip(other.m).map{
          case (arr1, arr2) => arr1.zip(arr2).map {
            case (num1, num2) => num1 * num2
          }
        }
        new Matrix(res, nrow, ncol)
      } else
        throw new IllegalArgumentException("Dimensions do not match!")
    }

    // matrix dot -- with another matrix
    def dot(other: Matrix): Matrix = {
      if (ncol == other.nrow) {

        val k = other.ncol
        val M = nrow

        // MapReduce matrix multiplication
        // map elements in the first matrix to pairs for further reduce
        val map1 = m.zipWithIndex.flatMap{
          case (row, rowID) => row.zipWithIndex.flatMap {
            case (num, colID) => for {j <- 0 until k} yield (rowID, j, colID.toLong) -> num
          }
        }
        // map elements in the second matrix to pairs for further reduce
        val map2 = other.m.zipWithIndex.flatMap{
          case (row, rowID) => row.zipWithIndex.flatMap {
            case (num, colID) => for {i <- 0 until M} yield (i.toLong, colID, rowID) -> num
          }
        }

        // join the mapped elements with key (i, j, k), each key gets exactly two
        // elements to multiply, then for each cell (i, j), reduce by summing
        // all values associated
        val allElem = map1.join(map2).map{
          case (k, v) => ((k._1, k._2), v._1 * v._2)}.reduceByKey(_ + _)

        // reconstruct the matrix form from value pairs ((i, j), value)
        val res = allElem.map{
          case ((row, col), num) => (row, (col, num))}.groupByKey.sortByKey().values.map(
          row => row.toArray.sortBy(_._1).map(_._2))

        new Matrix(res, nrow, other.ncol)
      } else
        throw new IllegalArgumentException("Dimensions do not match")
    }

    // matrix dot -- overload with Array[Array[Double]]
    def dot(other: Array[Array[Double]]): Matrix = {
      if (ncol == other.size) {
        val K = other.head.size
        val o = other.transpose

        val res = m.map(row =>
          o.map(orow =>
            orow.zip(row).map(pair =>
              pair._1 * pair._2).reduce(_ + _)) )
        new Matrix(res, nrow, K)
      } else
        throw new IllegalArgumentException("Dimensions do not match")
    }


    override def toString = {
      // serialize before converting to string to preserve
      // proper row order in conversion (if implemented with RDD
      // reduce, row order is not guaranteed)
      m.collect.map("[" + _.mkString(", ") + "]").mkString("\n")
      //m.map("[" + _.mkString(", ") + "]").reduce(_ + '\n' + _)
    }
  }



  class shallowNNclassifier(val hidden_dim: Int = 10,
                            val learn_rate: Double = 0.5,
                            val reg: Double = 0.0,
                            val max_iter: Int = 1000) {

    // data members of classifier object
    // since we will iteratively train our neural network
    // assign neural net parameters as mutable variables
    var W1 = Array(Array(0.0))
    var b1 = Array(Array(0.0))
    var W2 = Array(Array(0.0))
    var b2 = Array(Array(0.0))

    // helper functions
    def log(x: Matrix): Matrix = {
      val lg = x.m.map(row => row.map(math.log(_)))
      new Matrix(lg, x.nrow, x.ncol)
    }
    def exp(x: Matrix): Matrix = {
      val ex = x.m.map(row => row.map(math.exp(_)))
      new Matrix(ex, x.nrow, x.ncol)
    }
    def sigmoid(x: Matrix): Matrix = {
      val sig = x.m.map(row => row.map(e => 1.0 / (1.0 + math.exp(-e))))
      new Matrix(sig, x.nrow, x.ncol)
    }
    def d_sigmoid(x: Matrix): Matrix = {
      val sig = x.m.map(row => row.map(e =>
        math.exp(-e) / math.pow((1.0 + math.exp(-e)), 2)))
      new Matrix(sig, x.nrow, x.ncol)
    }
    def rand_mat(nrow: Int, ncol: Int): Array[Array[Double]] = {
      val rg = new Random()
      val mat =
        for {i <- 1 to nrow} yield
          { val row = for {j <- 1 to ncol} yield { rg.nextGaussian() }
            row.toArray
          }
      mat.toArray
    }
    // helper function for array matrix operation
    def matScal(mat: Array[Array[Double]], scale: Double): Array[Array[Double]] = {
      mat.map(row => row.map(_ * scale))
    }
    def matAdd(m1: Array[Array[Double]], m2: Array[Array[Double]]): Array[Array[Double]] = {
      if (m1.size == m2.size && m1.head.size == m2.head.size)
        m1.zip(m2).map{
          case (row1, row2) => row1.zip(row2).map(pair => pair._1 + pair._2)
        }
      else
        throw new IllegalArgumentException("Dimensions do not match!")
    }


    // initialize parameters (sideeffect)
    def init_params(D: Int, H: Int, K: Int): Unit = {
      // Initialize network parameters
      W1 = rand_mat(D, H)
      W2 = rand_mat(H, K)
      b1 = Array(Array.fill(H)(0))
      b2 = Array(Array.fill(K)(0))
      //(W2, b2, W1, b1)
    }

    def calculate_prob(x: Matrix): (Matrix, Matrix) = {
      // forward propagation to calculate class probabilities
      val z1 = x.dot(W1) + b1
      val hidden = sigmoid(z1)
      val scores = exp(hidden.dot(W2) + b2) // element-wise exponential

      // Softmax: normalize scores to make probabilities
      val sum = scores.m.map(row => row.sum)
      (new Matrix( scores.m.zip(sum).map(
        pair => pair._1.map(s => s / pair._2)), scores.nrow, scores.ncol ),
        hidden)
    }

    def calculate_loss(probs: Matrix, y: Matrix): Double = {
      // y is input labels with one-hot encoding
      val logp = log(probs)
      val entropy = logp * y
      val ce = -(entropy.m.map(row => row.sum).sum) / probs.nrow

      //println('\n' + logp.toString)
      //println('\n' + entropy.toString)

      // regularization terms
      val p1 = W1.map(row => row.map(elem => elem*elem).sum).sum
      val p2 = W2.map(row => row.map(elem => elem*elem).sum).sum

      ce + reg*p1 + reg*p2
    }

    def calculate_gradient(probs: Matrix, x: Matrix, y: Matrix, hidden: Matrix) = {
      // back-propagation of error
      // y is input labels with one-hot encoding
      val yerr = probs + (y * -1)

      val N = x.nrow
      // gradient of output layer
      // serialize to fit data type
      val dW2 = (hidden.transpose.dot(yerr) * (1.0 / N)).m.collect
      val db2 = matScal(Array(yerr.transpose.m.map(row => row.sum).collect), 1.0 / N)

      // gradient of hidden layer
      val dhidden = yerr.dot(W2.transpose) * d_sigmoid(x.dot(W1) + b1)
      val dW1 = (x.transpose.dot(dhidden) * (1.0 / N)).m.collect
      val db1 = matScal(Array(dhidden.transpose.m.map(row => row.sum).collect), 1.0 / N)

      val dW2p = matAdd(dW2, matScal(W2, reg))
      val dW1p = matAdd(dW1, matScal(W1, reg))
      //(dW2 + (W2 * reg), db2, dW1 + (W1 * reg), db1)
      (dW2p, db2, dW1p, db1)
    }

    def update_params(dW2: Array[Array[Double]],
                      db2: Array[Array[Double]],
                      dW1: Array[Array[Double]],
                      db1: Array[Array[Double]]): Unit = {

      W2 = matAdd(W2, matScal(dW2, -learn_rate))
      b2 = matAdd(b2, matScal(db2, -learn_rate))
      W1 = matAdd(W1, matScal(dW1, -learn_rate))
      b1 = matAdd(b1, matScal(db1, -learn_rate))

    }

    def fit(x: Matrix, y: Matrix): Unit = {
      init_params(x.ncol, hidden_dim, y.ncol)

      // iteratively train neural network
      for (rd <- 1 to max_iter) {
        val (probs, hidden) = calculate_prob(x)
        println(calculate_loss(probs, y))
        val (dW2, db2, dW1, db1) = calculate_gradient(probs, x, y, hidden)
        update_params(dW2, db2, dW1, db1)
      }
    }

    def predict(x: Matrix): RDD[Int] = {
      val (probs, _) = calculate_prob(x)
      probs.m.map(row => row.zipWithIndex.maxBy(pair => pair._1)._2)
    }

  }








  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Matrix Test")
    val spark = new SparkContext(conf)

    val a = Array(1.0, 2.0, 3.0, 4.0, 5.0)
    val b = Array(1.0, 0.0, 1.0, 0.0, 1.0)
    val c = Array(6.0, 7.0, 8.0, 9.0, 10.0)

    val m1 = new Matrix(spark.parallelize(Array(a, b, c)), 3, 5)
    val m2 = new Matrix(spark.parallelize(Array(c, a, b)), 3, 5)
    println('\n' + m1.toString)
    println('\n' + m2.toString)
    val m6 = m1 * m2
    println('\n' + m6.toString)
    val m7 = m1 + m2
    println('\n' + m7.toString)

    val m3 = m2.transpose
    println('\n' + m3.toString)
    //m1.shape
    //m3.shape
    val m8 = m1.dot(m3)
    println('\n' + m8.toString)

    val m5 = new Matrix(spark.parallelize(Array(a)), 1, 5)
    println('\n' + (m1 + m5).toString)

    val m9 = Array(c, a, b).transpose
    println('\n' + m1.dot(m9).toString)

    val m10 = Array(c)
    println('\n' + (m1 + m10).toString)

    val d = Array(1.0, 0.0, 0.0)
    val e = Array(0.0, 1.0, 0.0)
    val f = Array(0.0, 0.0, 1.0)

    val m11 = new Matrix(spark.parallelize(Array(d, e, f)), 3, 3)

    val x = spark.textFile("spiral_x.txt").map(row => row.split('\t').map(_.toDouble))
    val y = spark.textFile("spiral_y.txt").map(row => row.split('\t').map(_.toDouble))
    val mx = new Matrix(x, 300, 2)
    val my = new Matrix(y, 300, 3)

    //println('\n' + mx.toString)
    //println('\n' + my.toString)

    val nn = new shallowNNclassifier(learn_rate = 1.0, max_iter = 1000)
    //val randm = nn.rand_mat(5, 10)
    //println(randm.map('[' + _.mkString(", ") + ']').reduce(_ + '\n' + _))
    nn.fit(mx, my)
    val pred = nn.predict(mx)
    println("\n[" + pred.collect.mkString(", ") + ']')

    // evaluate prediction accuracy
    val acc = y.zip(pred).map{
      case (row, label) => row(label)
    }.reduce(_ + _) / 300.0
    println('\n' + acc.toString)

    spark.stop()
  }
}
