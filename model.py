import tensorflow as tf
import numpy as np
from numpy import random
import math




class Model():

  learn_rate = 0.01

  def __init__(self, number_of_samples, number_of_languages, number_of_features, number_of_relatedness_bins, number_of_distance_bins):
    self.sess = tf.Session()

    self.input = tf.placeholder(tf.float32, [None, 1, number_of_languages, number_of_features])  
    self.output = tf.placeholder(tf.float32, [None, number_of_samples, 1, number_of_features])
    
    self.relatedness_array = tf.placeholder(tf.float32, [1, number_of_samples, number_of_languages, number_of_relatedness_bins])
    self.distance_array = tf.placeholder(tf.float32, [1, number_of_samples, number_of_languages, number_of_distance_bins])

    self.relatedness_array_reshaped = tf.reshape(self.relatedness_array, [1, number_of_samples, number_of_languages, 1, number_of_relatedness_bins])
    self.distance_array_reshaped = tf.reshape(self.distance_array, [1, number_of_samples, number_of_languages, 1, number_of_distance_bins])

    self.relatedness_weights = tf.get_variable(name='relatedness_weights', dtype=tf.float32, shape=[number_of_features, number_of_relatedness_bins], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.distance_weights = tf.get_variable(name='distance_weights', dtype=tf.float32, shape=[number_of_features, number_of_distance_bins], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    
    self.r_1 = tf.reshape(self.relatedness_weights, [1, 1, 1, number_of_features, number_of_relatedness_bins])
    self.d_1 = tf.reshape(self.distance_weights, [1, 1, 1, number_of_features, number_of_distance_bins])
    
    self.r_2 = self.r_1 * self.relatedness_array_reshaped
    self.d_2 = self.d_1 * self.distance_array_reshaped
    
    self.r_3 = tf.reduce_sum(self.r_2, axis=4)
    self.d_3 = tf.reduce_sum(self.d_2, axis=4)
    
    self.r_final = self.r_3
    self.d_final = self.d_3
    
    self.intercept = tf.get_variable(name='intercept', dtype = tf.float32, shape = [1, 1, 1, number_of_features],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
#     self.borrowability = tf.get_variable(name='borrowability', dtype = tf.float32, shape = [1, 1, 1, number_of_features],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))

    self.borrowability = 0.01
    
    self.prediction_1 = (1.0 - ((1.0 - self.r_final) * (1.0 - self.borrowability))) * self.input
    self.prediction_1 = self.prediction_1 + (((1.0 - self.r_final) * (1.0 - self.borrowability)) * self.intercept)
    self.prediction_2 = (self.r_final * self.input) + ((1.0 - self.r_final) * self.intercept)

    self.actual = self.output
    self.loss_1 = 1.0 - tf.abs(self.actual - self.prediction_1)
    self.loss_2 = 1.0 - tf.abs(self.actual - self.prediction_2)

    self.loss_1 = self.loss_1 * self.d_final
    self.loss_2 = self.loss_2 * (1.0 - self.d_final)
    
    self.check_1 = self.loss_1 + self.loss_2
  
    self.loss_1 = tf.log(self.loss_1)
    self.loss_2 = tf.log(self.loss_2)
  
    self.na_array_1 = tf.placeholder(tf.float32, [1, number_of_samples, 1, number_of_features])
    self.na_array_2 = tf.placeholder(tf.float32, [1, 1, number_of_languages, number_of_features])

    self.loss_1 = self.loss_1 * self.na_array_1
    self.loss_1 = self.loss_1 * self.na_array_2
    self.loss_2 = self.loss_2 * self.na_array_1
    self.loss_2 = self.loss_2 * self.na_array_2
    
    self.loss_1 = tf.reduce_sum(self.loss_1, axis=3)
    self.loss_2 = tf.reduce_sum(self.loss_2, axis=3)

    
   
    
#     self.numerical_stability_constant = tf.reduce_max(self.loss_1) * -1.0

    self.numerical_stability_constant = 0
    self.loss_1 = tf.exp(self.loss_1 + self.numerical_stability_constant)
    self.loss_2 = tf.exp(self.loss_2 + self.numerical_stability_constant)

    self.moose_1 = self.loss_1
    self.moose_2 = self.loss_2 

    self.loss = self.loss_1 + self.loss_2
    
    
    
    self.moose = self.loss
    
    self.loss = tf.log(self.loss) - self.numerical_stability_constant

    self.total_loss = tf.reduce_mean(self.loss) * -1.0

  def train(self, input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, steps=200):
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss) 
    self.clip_op_1 = tf.assign(self.relatedness_weights, tf.clip_by_value(self.relatedness_weights, 0, 0.999))
    self.clip_op_2 = tf.assign(self.distance_weights, tf.clip_by_value(self.distance_weights, 0.001, 0.999))
    self.clip_op_3 = tf.assign(self.intercept, tf.clip_by_value(self.intercept, 0.01, 0.99))
#     self.clip_op_4 = tf.assign(self.borrowability, tf.clip_by_value(self.borrowability, 0.01, 0.99))   
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    self.feed = {self.input: input_array, self.output: output_array, self.na_array_1: na_array_1, 
    self.na_array_2: na_array_2, self.relatedness_array: relatedness_array, self.distance_array: distance_array}
    for i in range(steps):  
      print("After %d iterations:" % i)
#       print(self.sess.run(tf.reduce_max(self.moose_1), feed_dict=self.feed))
#       print(self.sess.run(tf.reduce_max(self.moose_2), feed_dict=self.feed))
# 
#       print(self.sess.run(tf.reduce_min(self.moose), feed_dict=self.feed))
#       print(self.sess.run(tf.reduce_max(self.moose), feed_dict=self.feed))
#       
#       print(self.sess.run(self.loss, feed_dict=self.feed))
#  
      print(self.sess.run(self.total_loss, feed_dict=self.feed))
#       print(self.sess.run(tf.reduce_min(self.check_1), feed_dict=self.feed))
#       print(self.sess.run(tf.reduce_max(self.check_1), feed_dict=self.feed))

#       print(self.sess.run(self.relatedness_weights))
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op_1)
      self.sess.run(self.clip_op_2)
      self.sess.run(self.clip_op_3)
#       self.sess.run(self.clip_op_4)

  def show_loss(self, input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array):
    self.feed = {self.input: input_array, self.output: output_array, self.na_array_1: na_array_1, 
    self.na_array_2: na_array_2, self.relatedness_array: relatedness_array, self.distance_array: distance_array}
    total_loss = self.sess.run(self.total_loss, feed_dict=self.feed)
#     print(total_loss)
    return total_loss

  def show_intercept(self):
    intercept = self.sess.run(self.intercept)
    print('Intercept: ', intercept)
    return intercept
  
  def show_weights(self):
    relatedness_weights = self.sess.run(self.relatedness_weights)
    print('Relatedness Weights: ', relatedness_weights)
    distance_weights = self.sess.run(self.distance_weights)
    print('Distance Weights: ', distance_weights)
    return relatedness_weights, distance_weights


