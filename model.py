import tensorflow as tf
import numpy as np
from numpy import random
import math



class Model():

  learn_rate = 0.01

  def __init__(self, number_of_simulations, number_of_samples, number_of_languages):
    self.sess = tf.Session()

    self.input = tf.placeholder(tf.float32, [None, 1, number_of_languages])
#     self.input_2 = tf.broadcast_to(self.input, [number_of_simulations, number_of_samples, number_of_languages])
    
  
    self.output = tf.placeholder(tf.float32, [None, number_of_samples, 1])
#     self.output_2 = tf.broadcast_to(self.output, [number_of_simulations, number_of_samples, number_of_languages])
  
    self.weights = tf.get_variable(name='weights', dtype = tf.float32, shape = [1, 1, number_of_languages],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
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
    learn_rate = 0.001
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
    loss = self.sess.run(self.total_loss, feed_dict=self.feed)
    print(loss)
    return loss

  def show_intercept(self):
    intercept = self.sess.run(self.intercept)
    print('Intercept: ', intercept)
    return intercept

class Autoencoder:  
  learn_rate = 0.01
  def __init__(self, simulated_feature_array, na_array, number_of_encoding_weights = 400, number_of_features=1, number_of_states=1):
    self.sess = tf.Session()
    self.simulated_feature_array = simulated_feature_array
    self.number_of_languages = np.shape(simulated_feature_array)[1]
    number_of_languages = self.number_of_languages
    number_of_simulations = simulated_feature_array.shape[0] 
    self.na_array = na_array
    self.encoder_input = tf.placeholder(tf.float32, [None, number_of_languages])
    self.encoder_weights = tf.get_variable(name='encoder_weights', dtype = tf.float32, shape = [number_of_languages, number_of_encoding_weights],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.encoder_weights_sum = tf.reduce_sum(self.encoder_weights, axis = 0)
    self.encoder_weights_sum = tf.broadcast_to(self.encoder_weights_sum, [number_of_languages, number_of_encoding_weights])
    self.encoder_weights_normalised = self.encoder_weights / self.encoder_weights_sum
    self.encoding = tf.matmul(self.encoder_input, self.encoder_weights_normalised)

    
    self.intercept = tf.get_variable(name='intercept', dtype=tf.float32, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))

    ''' going to comment these out'''
#     self.intercept_weights = tf.get_variable(name='intercept_weights', dtype=tf.float32, shape=[1,number_of_encoding_weights], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
#     self.encoding_after_weights = self.encoding * (1.0 - self.intercept_weights)
#     self.intercept_after_weights = self.intercept_weights * self.intercept
#     self.encoding = self.encoding_after_weights + self.intercept_after_weights    
   
    self.decoder_weights = tf.get_variable(name='decoder_weights', dtype = tf.float32, shape = [number_of_encoding_weights, number_of_languages],  initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))   
    self.decoder_weights_sum = tf.reduce_sum(self.decoder_weights, axis = 0)
    self.decoder_weights_sum = tf.broadcast_to(self.decoder_weights_sum, [number_of_encoding_weights, number_of_languages])
    self.decoder_weights_normalised = self.decoder_weights / self.decoder_weights_sum
    
    self.prediction = tf.matmul(self.encoding, self.decoder_weights_normalised)
    
    '''
    want a final intercept
    '''
    
    self.final_intercept_weighting = tf.get_variable(name='final_intercept', dtype=tf.float32, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.prediction = (self.prediction * (1 - self.final_intercept_weighting)) + (self.intercept * self.final_intercept_weighting)
    
    self.actual = self.encoder_input
    self.loss = tf.log(1.0 - tf.abs(self.actual - self.prediction))
    self.total_loss = tf.reduce_mean(self.loss) * -1.0
    
  def train(self, steps=10000):
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.encoder_clip = tf.assign(self.encoder_weights, tf.clip_by_value(self.encoder_weights, 0, 1))
    self.decoder_clip = tf.assign(self.decoder_weights, tf.clip_by_value(self.decoder_weights, 0, 1))
    self.intercept_clip = tf.assign(self.intercept, tf.clip_by_value(self.intercept, 0, 1))
#     self.intercept_weights_clip = tf.assign(self.intercept_weights, tf.clip_by_value(self.intercept_weights, 0, 1))
    self.final_intercept_weighting_clip = tf.assign(self.final_intercept_weighting, tf.clip_by_value(self.final_intercept_weighting, 0, 1))
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    self.feed = {self.encoder_input: self.simulated_feature_array}
    for i in range(steps):  
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.encoder_clip)
      self.sess.run(self.decoder_clip)
      self.sess.run(self.intercept_clip)
#       self.sess.run(self.intercept_weights_clip)
      self.sess.run(self.final_intercept_weighting_clip)
      print("After %d iterations:" % i)
      print(self.sess.run(self.prediction, feed_dict=self.feed))
      print(self.sess.run(self.actual, feed_dict=self.feed))
      print(self.sess.run(self.total_loss, feed_dict=self.feed))
#       print(self.sess.run(self.intercept))
#       print(self.sess.run(self.final_intercept_weighting))
#       print(self.sess.run(tf.reduce_min(self.encoding), feed_dict=self.feed))
#       print(self.sess.run(self.encoder_weights, feed_dict=self.feed))

  def show_loss(self, test_data):
    feed = {self.encoder_input: test_data}
    loss = self.sess.run(self.total_loss, feed_dict= feed)
    print('Loss: ', loss)
    return loss