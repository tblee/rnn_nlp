package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/26/16.
  */

import scala.math
import scala.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.math._
import breeze.numerics

object char_RNN_2layer {
  class char_RNN(val input: RDD[String],
                 hidden_dim_in: Int = 25,
                 seq_len_in: Int = 25,
                 learn_rate_in: Double = 0.1,
                 lim_in: Double = 5.0) {

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
    val hidden_dim: Int = hidden_dim_in
    val seq_len: Int = seq_len_in
    val learn_rate: Double = learn_rate_in
    val lim: Double = lim_in
    println(s"Input data has vocabulary size $vocab_size, " +
      s"initializing network with $hidden_dim hidden units")

    // initialize model parameters
    // define as Breeze matrices and vectors
    var Wxh1 = randGaussian(hidden_dim, vocab_size)
    var Whh1 = randGaussian(hidden_dim, hidden_dim)
    var Why1 = randGaussian(vocab_size, hidden_dim)
    var Wxh2 = randGaussian(hidden_dim, vocab_size)
    var Whh2 = randGaussian(hidden_dim, hidden_dim)
    var Why2 = randGaussian(vocab_size, hidden_dim)
    var bh1 = DenseVector.zeros[Double](hidden_dim)
    var by1 = DenseVector.zeros[Double](vocab_size)
    var bh2 = DenseVector.zeros[Double](hidden_dim)
    var by2 = DenseVector.zeros[Double](vocab_size)

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


    def step(inputs: Array[Int],
             targets: Array[Int],
             hprev: Array[DenseVector[Double]]) = {

      // in each step we feed our RNN model with a substring from corpus with
      // a specific length defined in seq_len, we do forward prop to obtain
      // output and loss, then back prop to compute gradient for parameter update

      // initialize I/O sequence
      val step_size = inputs.size
      val xt = new Array[DenseVector[Double]](step_size)
      val y1t = new Array[DenseVector[Double]](step_size)
      val h1t = new Array[DenseVector[Double]](step_size)
      val y2t = new Array[DenseVector[Double]](step_size)
      val h2t = new Array[DenseVector[Double]](step_size)
      val pt = new Array[DenseVector[Double]](step_size)
      var loss: Double = 0
      // forward pass
      for (t <- 0 until step_size) {
        // convert input into one-hot encoding
        val x = DenseVector.zeros[Double](vocab_size)
        x( inputs(t) ) = 1.0
        xt(t) = x

        // compute first hidden layer value
        val hp1 = if (t == 0) hprev(0) else h1t(t-1)
        h1t(t) = breeze.numerics.tanh(Wxh1 * xt(t) + Whh1 * hp1 + bh1)

        // compute second hidden layer value
        y1t(t) = Why1 * h1t(t) + by1
        val hp2 = if (t == 0) hprev(1) else h2t(t-1)
        h2t(t) = breeze.numerics.tanh(Wxh2 * y1t(t) + Whh2 * hp2 + bh2)

        // compute output vector
        y2t(t) = Why2 * h2t(t) + by2
        val expy = breeze.numerics.exp(y2t(t))
        pt(t) = expy / breeze.linalg.sum(expy)

        // compute and accumulate
        loss += -math.log( pt(t)(targets(t)) )
      }

      // back propagation
      var dWxh1 = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var dWhh1 = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var dWhy1 = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var dWxh2 = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var dWhh2 = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var dWhy2 = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var dbh1 = DenseVector.zeros[Double](hidden_dim)
      var dby1 = DenseVector.zeros[Double](vocab_size)
      var dbh2 = DenseVector.zeros[Double](hidden_dim)
      var dby2 = DenseVector.zeros[Double](vocab_size)
      var dhprev1 = DenseVector.zeros[Double](hidden_dim)
      var dhprev2 = DenseVector.zeros[Double](hidden_dim)
      for (t <- step_size-1 to 0 by -1) {
        val dy2 = pt(t)
        dy2( targets(t) ) -= 1.0

        // second layer
        dWhy2 += dy2 * (h2t(t).t)
        dby2 += dy2

        val dh2 = (Why2.t * dy2) + dhprev2
        val dhraw2 = (1.0 - (h2t(t) :* h2t(t))) :* dh2

        dbh2 += dhraw2
        dWxh2 += dhraw2 * (y1t(t).t)
        if (t > 0) dWhh2 += dhraw2 * (h2t(t-1).t)
        else dWhh2 += dhraw2 * hprev(1).t
        dhprev2 = Whh2 * dhraw2

        // first layer
        val dy1 = Wxh2.t * dhraw2

        dWhy1 += dy1 * (h1t(t).t)
        dby1 += dy1

        val dh1 = (Why1.t * dy1) + dhprev1
        val dhraw1 = (1.0 - (h1t(t) :* h1t(t))) :* dh1

        dbh1 += dhraw1
        dWxh1 += dhraw1 * (xt(t).t)
        if (t > 0) dWhh1 += dhraw1 * (h1t(t-1).t)
        else dWhh1 += dhraw1 * hprev(0).t
        dhprev1 = Whh1 * dhraw1
      }

      // clip gradient to prevent gradient vanishing or explosion
      // return loss, clipped gradient and the last hidden state
      (loss, clip(dWxh1), clip(dWhh1), clip(dWhy1), clip(dby1), clip(dbh1), h1t(step_size-1),
        clip(dWxh2), clip(dWhh2), clip(dWhy2), clip(dby2), clip(dbh2), h2t(step_size-1))
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
      var h1 = hprev(0)
      var h2 = hprev(1)
      x( input ) = 1.0
      for (t <- 0 until n) yield {

        // compute hidden layer values
        h1 = breeze.numerics.tanh(Wxh1 * x + Whh1 * h1 + bh1)
        val y1 = Why1 * h1 + by1
        h2 = breeze.numerics.tanh(Wxh2 * y1 + Whh2 * h2 + bh2)

        // compute output vector
        val y2 = Why2 * h2 + by2
        val expy = breeze.numerics.exp(y2)
        val id = sample(expy / breeze.linalg.sum(expy))
        //val id = breeze.linalg.argmax(expy)
        //out(t) = id2char(id)

        // put current output as next input
        x = DenseVector.zeros[Double](vocab_size)
        x( id ) = 1.0

        id2char(id)
      }

    }


    def fit() = {
      // fit the given RNN model

      // in this first version we serialize the training corpus
      val corpus = char_seq.collect
      val corpus_size = corpus.size
      var cur = 0
      var hprev = Array(DenseVector.zeros[Double](hidden_dim), DenseVector.zeros[Double](hidden_dim))
      var iter: Int = 0
      var smoothloss: Double = -math.log(1.0 / vocab_size) * seq_len

      // Adagrad parameters
      var mWxh1 = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var mWhh1 = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var mWhy1 = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var mbh1 = DenseVector.zeros[Double](hidden_dim)
      var mby1 = DenseVector.zeros[Double](vocab_size)
      var mWxh2 = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var mWhh2 = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var mWhy2 = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var mbh2 = DenseVector.zeros[Double](hidden_dim)
      var mby2 = DenseVector.zeros[Double](vocab_size)

      // gradient descent parameter update subroutine with Adagrad
      def update_param(dWxh1: DenseMatrix[Double],
                       dWhh1: DenseMatrix[Double],
                       dWhy1: DenseMatrix[Double],
                       dby1: DenseVector[Double],
                       dbh1: DenseVector[Double],
                       dWxh2: DenseMatrix[Double],
                       dWhh2: DenseMatrix[Double],
                       dWhy2: DenseMatrix[Double],
                       dby2: DenseVector[Double],
                       dbh2: DenseVector[Double]): Unit = {

        // Adagrad step
        mWxh1 += dWxh1 :* dWxh1
        mWhh1 += dWhh1 :* dWhh1
        mWhy1 += dWhy1 :* dWhy1
        mbh1 += dbh1 :* dbh1
        mby1 += dby1 :* dby1
        mWxh2 += dWxh2 :* dWxh2
        mWhh2 += dWhh2 :* dWhh2
        mWhy2 += dWhy2 :* dWhy2
        mbh2 += dbh2 :* dbh2
        mby2 += dby2 :* dby2
        Wxh1 -= learn_rate * (dWxh1 :/ breeze.numerics.sqrt(mWxh1 + 1e-8))
        Whh1 -= learn_rate * (dWhh1 :/ breeze.numerics.sqrt(mWhh1 + 1e-8))
        Why1 -= learn_rate * (dWhy1 :/ breeze.numerics.sqrt(mWhy1 + 1e-8))
        by1 -= learn_rate * (dby1 :/ breeze.numerics.sqrt(mby1 + 1e-8))
        bh1 -= learn_rate * (dbh1 :/ breeze.numerics.sqrt(mbh1 + 1e-8))
        Wxh2 -= learn_rate * (dWxh2 :/ breeze.numerics.sqrt(mWxh2 + 1e-8))
        Whh2 -= learn_rate * (dWhh2 :/ breeze.numerics.sqrt(mWhh2 + 1e-8))
        Why2 -= learn_rate * (dWhy2 :/ breeze.numerics.sqrt(mWhy2 + 1e-8))
        by2 -= learn_rate * (dby2 :/ breeze.numerics.sqrt(mby2 + 1e-8))
        bh2 -= learn_rate * (dbh2 :/ breeze.numerics.sqrt(mbh2 + 1e-8))
      }

      // iterativesly train RNN model with fixed-size sequence
      while (cur >= 0) { // artificial condition for infinite training loop
        // reset training cycle when reaching end of corpus
        if (cur+seq_len+1 > corpus_size) {
          cur = 0
          hprev = Array(DenseVector.zeros[Double](hidden_dim),
            DenseVector.zeros[Double](hidden_dim))
        }
        //println(hprev)

        //val endpt = min(cur+seq_len+1, corpus_size)
        val inputs = corpus.slice(cur, cur+seq_len).map(char => char2id(char))
        val targets = corpus.slice(cur+1, cur+seq_len+1).map(char => char2id(char))

        // make a step in training
        val (loss, dWxh1, dWhh1, dWhy1, dby1, dbh1, h1, dWxh2, dWhh2, dWhy2, dby2, dbh2, h2) =
          step(inputs, targets, hprev)

        // update parameters
        update_param(dWxh1, dWhh1, dWhy1, dby1, dbh1, dWxh2, dWhh2, dWhy2, dby2, dbh2)
        smoothloss = 0.999 * smoothloss + 0.001 * loss

        // with certain checkpoint, output current loss and and example paragraph
        // generated by the RNN model
        if (iter % 100 == 0) {
          println(s"Training loss at iteration $iter: $smoothloss")
          println(transform(char2id(corpus(cur)), hprev, 200).mkString("") + "\n")
        }

        // update training cycle and memory
        hprev = Array(h1, h2)
        cur += seq_len
        iter += 1
      }

    }

  }



  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("char_RNN")
    val spark = new SparkContext(conf)

    // read input corpus
    val data = spark.textFile("min-char-rnn-test.txt")

    // create and fit char-RNN model with corpus
    val rnn = new char_RNN(data)
    rnn.fit()

  }
}
