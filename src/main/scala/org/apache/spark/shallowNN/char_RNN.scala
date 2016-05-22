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
    val char_seq = input.flatMap(row => row.split(" ")).flatMap(word => word.toCharArray)

    // make vocabulary maps
    val vocab = char_seq.distinct.zipWithIndex
    val char2id = vocab.collect.toMap
    val id2char = vocab.map{case (char, id) => (id, char)}.collect.toMap

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




  }








  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("char_RNN")
    val spark = new SparkContext(conf)

    // read input corpus
    val data = spark.textFile("min-char-rnn-test.txt")
    //val word_seq = data.flatMap(row => row.split(" "))

    val rnn = new char_RNN(data)

    // make a dictionary from input document
    //val vocab = word_seq.distinct.zipWithIndex
    //val word2id = vocab.collect.toMap
    //val id2word = vocab.map{case (word, id) => (id, word)}.collect.toMap


  }
}
