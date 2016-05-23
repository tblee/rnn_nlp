package org.apache.spark.shallowNN

/**
  * Created by tblee on 5/21/16.
  */

import scala.math
import scala.util.Random
import org.apache.spark._
import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.math._
import breeze.numerics


object char_RNN {

  class char_RNN(val input: RDD[String],
                 hidden_dim_in: Int = 100,
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
    var Wxh = randGaussian(hidden_dim, vocab_size)
    var Whh = randGaussian(hidden_dim, hidden_dim)
    var Why = randGaussian(vocab_size, hidden_dim)
    var bh = DenseVector.zeros[Double](hidden_dim)
    var by = DenseVector.zeros[Double](vocab_size)

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
             hprev: DenseVector[Double]) = {

      // in each step we feed our RNN model with a substring from corpus with
      // a specific length defined in seq_len, we do forward prop to obtain
      // output and loss, then back prop to compute gradient for parameter update

      // initialize I/O sequence
      val step_size = inputs.size
      val xt = new Array[DenseVector[Double]](step_size)
      val yt = new Array[DenseVector[Double]](step_size)
      val ht = new Array[DenseVector[Double]](step_size)
      val pt = new Array[DenseVector[Double]](step_size)
      var loss: Double = 0
      // forward pass
      for (t <- 0 until step_size) {
        // convert input into one-hot encoding
        val x = DenseVector.zeros[Double](vocab_size)
        x( inputs(t) ) = 1.0
        xt(t) = x

        // compute hidden layer value
        val hp = if (t == 0) hprev else ht(t-1)
        ht(t) = breeze.numerics.tanh(Wxh * xt(t) + Whh * hp + bh)

        // compute output vector
        yt(t) = Why * ht(t) + by
        val expy = breeze.numerics.exp(yt(t))
        pt(t) = expy / breeze.linalg.sum(expy)

        // compute and accumulate
        loss += -math.log( pt(t)(targets(t)) )
      }

      // back propagation
      var dWxh = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var dWhh = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var dWhy = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var dbh = DenseVector.zeros[Double](hidden_dim)
      var dby = DenseVector.zeros[Double](vocab_size)
      var dhprev = DenseVector.zeros[Double](hidden_dim)
      for (t <- step_size-1 to 0 by -1) {
        val dy = pt(t)
        dy( targets(t) ) -= 1.0

        dWhy += dy * (ht(t).t)
        dby += dy

        val dh = (Why.t * dy) + dhprev
        val dhraw = (1.0 - (ht(t) :* ht(t))) :* dh

        dbh += dhraw
        dWxh += dhraw * (xt(t).t)
        if (t > 0) dWhh += dhraw * (ht(t-1).t)
        else dWhh += dhraw * hprev.t
        dhprev = Whh * dhraw
      }

      // clip gradient to prevent gradient vanishing or explosion
      // return loss, clipped gradient and the last hidden state
      (loss, clip(dWxh), clip(dWhh), clip(dWhy), clip(dby), clip(dbh), ht(step_size-1))
    }

    def transform(input: Int = 0,
                  hprev: DenseVector[Double] = DenseVector.zeros[Double](hidden_dim),
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
      var h = hprev
      x( input ) = 1.0
      for (t <- 0 until n) yield {

        // compute hidden layer value
        h = breeze.numerics.tanh(Wxh * x + Whh * h + bh)

        // compute output vector
        val y = Why * h + by
        val expy = breeze.numerics.exp(y)
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
      var hprev = DenseVector.zeros[Double](hidden_dim)
      var iter: Int = 0
      var smoothloss: Double = -math.log(1.0 / vocab_size) * seq_len

      // Adagrad parameters
      var mWxh = DenseMatrix.zeros[Double](hidden_dim, vocab_size)
      var mWhh = DenseMatrix.zeros[Double](hidden_dim, hidden_dim)
      var mWhy = DenseMatrix.zeros[Double](vocab_size, hidden_dim)
      var mbh = DenseVector.zeros[Double](hidden_dim)
      var mby = DenseVector.zeros[Double](vocab_size)

      // gradient descent parameter update subroutine with Adagrad
      def update_param(dWxh: DenseMatrix[Double],
                       dWhh: DenseMatrix[Double],
                       dWhy: DenseMatrix[Double],
                       dby: DenseVector[Double],
                       dbh: DenseVector[Double]): Unit = {

        // Adagrad step
        mWxh += dWxh :* dWxh
        mWhh += dWhh :* dWhh
        mWhy += dWhy :* dWhy
        mbh += dbh :* dbh
        mby += dby :* dby
        Wxh -= learn_rate * (dWxh :/ breeze.numerics.sqrt(mWxh + 1e-8))
        Whh -= learn_rate * (dWhh :/ breeze.numerics.sqrt(mWhh + 1e-8))
        Why -= learn_rate * (dWhy :/ breeze.numerics.sqrt(mWhy + 1e-8))
        by -= learn_rate * (dby :/ breeze.numerics.sqrt(mby + 1e-8))
        bh -= learn_rate * (dbh :/ breeze.numerics.sqrt(mbh + 1e-8))
      }

      // iterativesly train RNN model with fixed-size sequence
      while (cur >= 0) { // artificial condition for infinite training loop
        // reset training cycle when reaching end of corpus
        if (cur+seq_len+1 > corpus_size) {
          cur = 0
          hprev = DenseVector.zeros[Double](hidden_dim)
        }
        //println(hprev)

        //val endpt = min(cur+seq_len+1, corpus_size)
        val inputs = corpus.slice(cur, cur+seq_len).map(char => char2id(char))
        val targets = corpus.slice(cur+1, cur+seq_len+1).map(char => char2id(char))

        // make a step in training
        val (loss, dWxh, dWhh, dWhy, dby, dbh, h) = step(inputs, targets, hprev)

        // update parameters
        update_param(dWxh, dWhh, dWhy, dby, dbh)
        smoothloss = 0.999 * smoothloss + 0.001 * loss

        // with certain checkpoint, output current loss and and example paragraph
        // generated by the RNN model
        if (iter % 100 == 0) {
          println(s"Training loss at iteration $iter: $smoothloss")
          println(transform(char2id(corpus(cur)), hprev, 200).mkString("") + "\n")
        }

        // update training cycle and memory
        hprev = h
        cur += seq_len
        iter += 1
      }

    }

  }



  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("char_RNN")
    val spark = new SparkContext(conf)

    // read input corpus
    val data = spark.textFile("min-char-rnn-test-tiny.txt")

    // create and fit char-RNN model with corpus
    val rnn = new char_RNN(data)
    rnn.fit()

  }
}
