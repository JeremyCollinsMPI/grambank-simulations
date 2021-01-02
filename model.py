import tensorflow as tf
import numpy as np
from numpy import random
import math




class Model():

  learn_rate = 0.01

  def __init__(self, number_of_simulations, number_of_samples, number_of_languages, number_of_relatedness_bins, number_of_distance_bins):
    self.sess = tf.Session()
    self.input = tf.placeholder(tf.float32, [None, 1, number_of_languages])  
    self.output = tf.placeholder(tf.float32, [None, number_of_samples, 1])
    self.relatedness_array = tf.placeholder(tf.float32, [1, number_of_samples, number_of_languages, number_of_relatedness_bins]
    self.distance_array = tf.placeholder(tf.float32, [1, number_of_samples, number_of_languages, number_of_distance_bins]
        
    self.relatedness_weights = tf.get_variable(name='relatedness_weights', dtype=tf.float32, shape=[1, number_of_relatedness_bins], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.distance_weights = tf.get_variable(name='distance_weights', dtype=tf.float32, shape=[1, number_of_distance_bins], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.intercept = tf.get_variable(name='intercept', dtype = tf.float32, shape = [1],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
 
    
 
    self.r_1 = tf.reshape(self.relatedness_weights, [1, 1, 1, number_of_relatedness_bins)
    self.d_1 = tf.reshape(self.distance_weights, [1, 1, 1, number_of_distance_bins)
    
    self.r_2 = self.relatedness_array * self.r_1
    self.d_2 = self.distance_array * self.d_1
    
    self.r_3 = tf.reduce_sum(self.r_2, axis=3)
    self.d_3 = tf.reduce_sum(self.d_2, axis=3)
    
    self.r_final = self.r_3
    self.d_final = self.d_3
    
    self.relatedness_or_contact_prediction = 1.0 - ((1.0 - self.r_final) * (1.0 - self.d_final) * self.input)
    self.intercept_prediction = (1.0 - self.r_final) * (1.0 - self.d_final) * self.intercept
    self.prediction = self.relatedness_or_contact_prediction + self.intercept_prediction
    
    '''
    r_final needs to be of shape [1, number_of_samples, number_of_languages]
    
    relatedness weights currently of shape [1, number of relatedness bins]
    you need to multiply the relatedness array by weights
    [1, samples, languages, number of relatedness bins]
    
    broadcast weights to [1,1,1, relatedness bins]
    
    multiply element wise
    
    
    shape [1, samples, languages, number of relatedness bins]
    
    then reduce sum along last axis
    
    shape [1, samples, languages]
    
    
    
    
    
    
    
    
    
    ((1 - (1-a)(1-b)) * input)  + ((1-a)(1-b) * intercept)
    
    
    
    
    
    
    you multiply the input element wise
    
    
    
   1 - (1-a)(1-b) is the probability of having a shared value between two languages,
  if you think of a and b as the probability of having a shared value due to relatedness and due to contact respectively
  
  then 1 - that * intercept.
  
   ????
   
   1
   
   1
   
   0.5
  
    0.7
    
    
    
    you have the probability a of sharing the value due to relatedness; the probability b of sharing the 
    value due to contact; and the intercept, which is the value assigned if after the two dice rolls neither relatedness
    nor contact has caused the second language to share the value of the first language
    
    so the expected value for language 2 is:
    
    sum over the different scenarios:
    
    i) shared value due to relatedness: 
       a * input
    
    ii) shared value due to contact:
      b * input
      
    iii) random value
      (1-a)(1-b) * intercept
      
    ^ not sure this is correct
    
    easier to simply find the value due to relatedness or contact.
    
    i) shared value due to relatedness or contact
      (1 - (1-a)(1-b)) * input
    
    ii) random value
      (1-a)(1-b) * intercept
    
    
    so the prediction is
    
    ((1 - (1-a)(1-b)) * input)  + ((1-a)(1-b) * intercept)
    
    '''





  
    self.weights = tf.get_variable(name='weights', dtype = tf.float32, shape = [1, number_of_samples, number_of_languages],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.intercept = tf.get_variable(name='intercept', dtype = tf.float32, shape = [1],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    
    self.prediction = (self.input * self.weights) + (self.intercept * (1 - self.weights))
    self.actual = self.output
    self.loss = tf.log(1.0 - tf.abs(self.actual - self.prediction))
    
    
    self.na_array_1 = tf.placeholder(tf.float32, [1, number_of_samples, 1])
    self.na_array_2 = tf.placeholder(tf.float32, [1, 1, number_of_languages])
    
    self.loss = self.loss * self.na_array_1
    self.loss = self.loss * self.na_array_2
    self.total_loss = tf.reduce_mean(self.loss) * -1.0

  def train(self, input_array, output_array, na_array_1, na_array_2, steps=200):
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss) 
    self.clip_op_1 = tf.assign(self.weights, tf.clip_by_value(self.weights, 0, 0.99))
    self.clip_op_2 = tf.assign(self.intercept, tf.clip_by_value(self.intercept, 0.01, 0.99))
    
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    self.feed = {self.input: input_array, self.output: output_array, self.na_array_1: na_array_1, self.na_array_2: na_array_2}
    for i in range(steps):  
      print("After %d iterations:" % i)
#       print(self.sess.run(self.prediction, feed_dict=self.feed))
#       print(self.sess.run(self.actual, feed_dict=self.feed))
      print(self.sess.run(self.total_loss, feed_dict=self.feed))
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op_1)
      self.sess.run(self.clip_op_2)
    
  def show_loss(self, input_array, output_array, na_array_1, na_array_2):
    self.feed = {self.input: input_array, self.output: output_array, self.na_array_1: na_array_1, self.na_array_2: na_array_2}
#     loss = self.sess.run(self.total_loss, feed_dict=self.feed)
    loss = self.sess.run(tf.reduce_mean(tf.reduce_mean(self.loss, axis=2), axis=1), feed_dict=self.feed)
    print(loss)
    return loss

  def show_intercept(self):
    intercept = self.sess.run(self.intercept)
    print('Intercept: ', intercept)
    return intercept
  
  def show_weights(self):
    weights = self.sess.run(self.weights)
    print('Weights: ', weights)
    return weights



