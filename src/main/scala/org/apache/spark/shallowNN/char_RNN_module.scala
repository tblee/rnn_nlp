package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/29/16.
  */

import scala.math
import scala.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.rdd.RDDFunctions._
import breeze.linalg._
import breeze.math._
import breeze.numerics

object char_RNN_module {
  class char_RNN(val input: RDD[String],
                 val num_layers: Int = 1,
                 val hidden_dim: Int = 25,
                 val seq_len: Int = 25,
                 val learn_rate: Double = 0.1,
                 val lim: Double = 5.0)
    extends Serializable {

    // Constructor of character RNN class
    // parse the input corpus and produce a vocabulary mapping to
    // map each character to a unique ID

    // convert input corpus to a sequence of words
    val char_seq = input.flatMap(word => word.toCharArray)

    // make vocabulary maps
    val vocab = char_seq.distinct.zipWithIndex
    val char2id = vocab.map{case (char, id) => (char, id.toInt)}.collect.toMap
    val id2char = vocab.map{case (char, id) => (id.toInt, char)}.collect.toMap

    // define and initialize model variables
    // basic and hyperparameters
    val vocab_size: Int = vocab.count.toInt

    println(s"Input data has vocabulary size $vocab_size, " +
      s"initializing network with $num_layers layers " +
      s"each has $hidden_dim hidden units")

    // initialize model parameters for modules
    var Win = Array.fill(num_layers){randGaussian(hidden_dim, vocab_size)}
    var Wh = Array.fill(num_layers){randGaussian(hidden_dim, hidden_dim)}
    var Wout = Array.fill(num_layers){randGaussian(vocab_size, hidden_dim)}
    var bh = Array.fill(num_layers){DenseVector.zeros[Double](hidden_dim)}
    var bout = Array.fill(num_layers){DenseVector.zeros[Double](vocab_size)}

    // Helper function to produce Breeze Matrix randomized
    // with Gaussian
    def randGaussian(nrow: Int, ncol: Int) = {
      val rg = new Random()
      DenseMatrix.zeros[Double](nrow, ncol).map(
        elem => rg.nextGaussian() * 0.01
      )
    }

    // helper function to clip Breeze matrix or vector values
    def clip(m: DenseVector[Double]): DenseVector[Double] = m.map {
      elem =>
        if (elem > lim) lim
        else if (elem < -lim) -lim
        else elem
    }
    def clip(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map{
      elem =>
        if (elem > lim) lim
        else if (elem < -lim) -lim
        else elem
    }


    def step(input_seq: Array[Int],
             hprev: Array[DenseVector[Double]]) = {

      // in each step we feed our RNN model with a substring from corpus with
      // a specific length defined in seq_len, we do forward prop to obtain
      // output and loss, then back prop to compute gradient for parameter update

      val inputs = input_seq.slice(0, seq_len)
      val targets = input_seq.slice(1, seq_len+1)

      // initialize I/O sequence
      val step_size = inputs.size
      val xt = new Array[DenseVector[Double]](step_size)

      val yt = Array.fill(num_layers){new Array[DenseVector[Double]](step_size)}
      val ht = Array.fill(num_layers){new Array[DenseVector[Double]](step_size)}

      val pt = new Array[DenseVector[Double]](step_size)
      var loss: Double = 0
      // forward pass
      for (t <- 0 until step_size) {
        // convert input into one-hot encoding
        val x = DenseVector.zeros[Double](vocab_size)
        x( inputs(t) ) = 1.0
        xt(t) = x

        // forward pass with layer modules
        for (layer <- 0 until num_layers) {
          // specify module input
          val xin = if (layer == 0) xt(t) else yt(layer-1)(t)

          val hp = if (t == 0) hprev(layer) else ht(layer)(t-1)
          ht(layer)(t) = breeze.numerics.tanh(Win(layer) * xin + Wh(layer) * hp + bh(layer))

          yt(layer)(t) = Wout(layer) * ht(layer)(t) + bout(layer)
        }
        val expy = breeze.numerics.exp(yt(num_layers - 1)(t))
        pt(t) = expy / breeze.linalg.sum(expy)

        // compute and accumulate
        loss += -math.log( pt(t)(targets(t)) )
      }

      // back propagation
      // initialize model parameters for modules
      val dWin = Array.fill(num_layers){DenseMatrix.zeros[Double](hidden_dim, vocab_size)}
      val dWh = Array.fill(num_layers){DenseMatrix.zeros[Double](hidden_dim, hidden_dim)}
      val dWout = Array.fill(num_layers){DenseMatrix.zeros[Double](vocab_size, hidden_dim)}
      val dbh = Array.fill(num_layers){DenseVector.zeros[Double](hidden_dim)}
      val dbout = Array.fill(num_layers){DenseVector.zeros[Double](vocab_size)}
      val dhprev = Array.fill(num_layers){DenseVector.zeros[Double](hidden_dim)}

      for (t <- step_size-1 to 0 by -1) {

        // module back-prop
        var delta = pt(t)
        delta( targets(t) ) -= 1.0
        for (layer <- (num_layers-1) to 0 by -1) {
          dWout(layer) += delta * ht(layer)(t).t
          dbout(layer) += delta

          val dh = (Wout(layer).t * delta) + dhprev(layer)
          val dhraw = (1.0 - (ht(layer)(t) :* ht(layer)(t))) :* dh

          dbh(layer) += dhraw
          if (layer > 0) dWin(layer) += dhraw * yt(layer - 1)(t).t
          else dWin(layer) += dhraw * xt(t).t

          if (t > 0) dWh(layer) += dhraw * ht(layer)(t - 1).t
          else dWh(layer) += dhraw * hprev(layer).t
          dhprev(layer) = Wh(layer) * dhraw

          // propage loss to previous layer
          delta = (Win(layer).t) * dhraw
        }

      }

      // clip gradient to prevent gradient vanishing or explosion
      // return loss, clipped gradient and the last hidden state
      (loss, dWin.map(clip(_)), dWh.map(clip(_)), dWout.map(clip(_)), dbout.map(clip(_)), dbh.map(clip(_)), 1)
    }

    def transform(input: Int = 0,
                  hprev: Array[DenseVector[Double]],
                  n: Int = seq_len) = {

      // the transform function takes an input to kick-start RNN model in
      // generating a sequence of output with specified length n.
      // previous hidden state can be provided, or it will be default to zero

      // helper function to take sample from a prob distribution
      def sample(dist: DenseVector[Double]): Int = {

        // assume the input distribution vector has length = vocab_size
        val accu = new Array[Double](vocab_size)
        accu(0) = dist(0)
        for (i <- 1 until vocab_size) accu(i) = accu(i-1) + dist(i)

        def bSearch(l: Int, r: Int, target: Float): Int = {
          if (target >= accu(r-1)) r
          else if (target < accu(l+1)) l+1
          else {
            val mid = (l + r) / 2
            if (target < accu(mid)) bSearch(l, mid, target)
            else bSearch(mid, r, target)
          }
        }

        // use binary search to find sampled id
        val d = Random.nextFloat()
        if (d <= accu(0)) 0
        else if (d >= accu(vocab_size-1)) vocab_size - 1
        else bSearch(0, vocab_size-1, d)
      }


      var x = DenseVector.zeros[Double](vocab_size)
      val h = hprev
      x( input ) = 1.0
      for (t <- 0 until n) yield {

        // forward pass with layer modules
        for (layer <- 0 until num_layers) {
          h(layer) = breeze.numerics.tanh(Win(layer) * x + Wh(layer) * h(layer) + bh(layer))
          x = Wout(layer) * h(layer) + bout(layer)
        }
        val expy = breeze.numerics.exp(x)
        val id = sample(expy / breeze.linalg.sum(expy))

        // put current output as next input
        x = DenseVector.zeros[Double](vocab_size)
        x( id ) = 1.0

        id2char(id)
      }

    }


    def fit() = {
      // fit the given RNN model
      var smoothloss: Double = -math.log(1.0 / vocab_size) * seq_len

      // Adagrad parameters for modules
      val mWin = Array.fill(num_layers){DenseMatrix.zeros[Double](hidden_dim, vocab_size)}
      val mWh = Array.fill(num_layers){DenseMatrix.zeros[Double](hidden_dim, hidden_dim)}
      val mWout = Array.fill(num_layers){DenseMatrix.zeros[Double](vocab_size, hidden_dim)}
      val mbh = Array.fill(num_layers){DenseVector.zeros[Double](hidden_dim)}
      val mbout = Array.fill(num_layers){DenseVector.zeros[Double](vocab_size)}

      // gradient descent parameter update subroutine with Adagrad
      def update_param(dWin: Array[DenseMatrix[Double]],
                       dWh: Array[DenseMatrix[Double]],
                       dWout: Array[DenseMatrix[Double]],
                       dbout: Array[DenseVector[Double]],
                       dbh: Array[DenseVector[Double]]): Unit = {

        // update parameter for layers
        for (layer <- 0 until num_layers) {
          mWin(layer) += dWin(layer) :* dWin(layer)
          mWh(layer) += dWh(layer) :* dWh(layer)
          mWout(layer) += dWout(layer) :* dWout(layer)
          mbout(layer) += dbout(layer) :* dbout(layer)
          mbh(layer) += dbh(layer) :* dbh(layer)

          Win(layer) -= learn_rate * (dWin(layer) :/ breeze.numerics.sqrt(mWin(layer) + 1e-8))
          Wh(layer) -= learn_rate * (dWh(layer) :/ breeze.numerics.sqrt(mWh(layer) + 1e-8))
          Wout(layer) -= learn_rate * (dWout(layer) :/ breeze.numerics.sqrt(mWout(layer) + 1e-8))
          bout(layer) -= learn_rate * (dbout(layer) :/ breeze.numerics.sqrt(mbout(layer) + 1e-8))
          bh(layer) -= learn_rate * (dbh(layer) :/ breeze.numerics.sqrt(mbh(layer) + 1e-8))
        }
      }

      // prepare training data as sliding windows
      val train2id = char_seq.map(c => char2id(c)).sliding(seq_len + 1)
      var epoch = 0

      // train loop to feed sliding window to RNN and update parameters
      // training with gradient descent, in each epoch, every sliding window is taken to calculate
      // gradient and an average gradient is obtained and used to update parameters after each epoch.'

      while (epoch >= 0) {
        val h_init = Array.fill(num_layers){DenseVector.zeros[Double](hidden_dim)}
        val gradient_seq = train2id.map(window => step(window, h_init))
        val gradients = gradient_seq.reduce{
          case (x, y) =>
            (x._1 + y._1,
              (x._2).zip(y._2).map{case (a, b) => a + b},
              (x._3).zip(y._3).map{case (a, b) => a + b},
              (x._4).zip(y._4).map{case (a, b) => a + b},
              (x._5).zip(y._5).map{case (a, b) => a + b},
              (x._6).zip(y._6).map{case (a, b) => a + b}, x._7 + y._7)
        }

        val (loss, dWin, dWh, dWout, dbout, dbh, count) = gradients

        // update parameters
        val ct = count.toDouble
        update_param(dWin.map(_ / ct), dWh.map(_ / ct), dWout.map(_ / ct), dbout.map(_ / ct), dbh.map(_ / ct))
        smoothloss = 0.999 * smoothloss + 0.001 * (loss / ct)

        println(s"Training loss at epoch $epoch: $loss")
        val h_kickoff = Array.fill(num_layers){DenseVector.rand[Double](hidden_dim)}
        println(transform(0 , h_kickoff, 200).mkString("") + "\n")

        epoch += 1
      }
    }
  }



  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("char_RNN")
    val spark = new SparkContext(conf)

    // read input corpus
    val data = spark.textFile("min-char-rnn-test.txt")

    // create and fit char-RNN model with corpus
    val rnn = new char_RNN(input = data,
      num_layers = 1,
      hidden_dim = 25,
      seq_len = 5,
      learn_rate = 0.1,
      lim = 5.0)
    rnn.fit()

  }
}
