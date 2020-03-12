# Version of the training and compression parts of Autoencoder Prototype 29 designed to run in Google Colaboratory
# Note: Does not support error-bounds

# %tensorflow_version 1.x
import array
import math
import numpy as np  
import os
import progressbar
import tensorflow as tf
import time
from google.colab import files

sess = tf.Session()

def get_data(file_name):
  data = f[file_name]
  record_bytes = len(data)
  print("Length of file:", len(data))
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

  # Convert the data to a vector of numbers and store it in data_num
  if(file_name[-4:] == ".txt"):
    data_num = data.decode().split('\n')
    if(data_num[len(data_num) - 1] == ""):
      data_num.remove("")
    for i in range(len(data_num)):
      data_num[i] = float(data_num[i])
    data_num = np.array(data_num)
  else:
    data_node = tf.placeholder(tf.string)
    record_bytes = tf.decode_raw(data_node, tf.float64)
    data_num = sess.run(record_bytes, {data_node: data})
  return data_num

def normalize_data(data_num):
  size = tf.size(data_num)  # Size of data_num (the vector of numbers)
  data_num_size = sess.run(size)
  print("Number of Data Points:", data_num_size)

  modifications = [1 for x in range(data_num_size)]  # Initialize modifications with ones

  # Ensure that all of the data is either 0 or [0.01, 0.1)
  # Note: Does not work with negative numbers
  modifications_all = [0] * data_num_size

  for i in range(data_num_size):
    try:
      modifications_all[i] = (math.floor(math.log10(data_num[i])) + 2) * -1
    except ValueError:
      modifications_all[i] = 0

  data_num = [data_num[i] * (10 ** modifications_all[i]) for i in range(data_num_size)]

  modifications = []
  strides = []
  index = -1
  last_num = 0.5
  mod_min = modifications_all[0]

  for i in range(data_num_size):
    if modifications_all[i] == last_num:
      strides[index] += 1
    else:
      modifications.append(modifications_all[i])
      strides.append(1)
      last_num = modifications_all[i]
      index += 1
      if(modifications_all[i] < mod_min):
        mod_min = modifications_all[i]

  if mod_min < 0:
    for i in range(index + 1):
      modifications[i] += (mod_min * -1)
  
  return data_num, data_num_size, modifications, strides, mod_min, index

# Training
print("TRAINING THE AUTOENCODER")
print("Please choose a file for training:")
f = files.upload()

name = ""

for f_i in f.keys():
  name = f_i

training_start = time.time()
optimizer_sum = 0.0
error_organizing_sum = 0.0

print("The name of the file for training is:", name)

data_num = get_data(name)
original_data_num = np.copy(data_num)  # Copy data_num (by value, not reference) and store the copy in original_data_num

normalize_data_method_start = time.time()
data_num, data_num_size, modifications, strides, mod_min, index = normalize_data(data_num)
normalize_data_method_end = time.time()

training_epochs = 10000
batch_size = 64
n_input = 256

display = 5000

# Error bound definitions
error_0 = 0    # if error = 0
error_A = 0    # if 0 < error < 0.0001 (0.01%)
error_B = 0    # if 0.0001 <= error < 0.001 (0.1%)
error_C = 0    # if 0.001 <= error < 0.01 (1%)
error_D = 0    # if 0.01 <= error < 0.1 (10%)
error_E = 0    # if 0.1 <= error < 1 (100%)
error_F = 0    # if 1 <= error
error_sum = 0

# TensorFlow Placeholders
X = tf.placeholder("float", [None, None])
Y = tf.placeholder("float", [None, n_input])
Z = tf.placeholder("float", [None, n_input])
dropout_prob = tf.placeholder("float")

'''
n_hidden_1 = int(n_input / 8)
n_hidden_2 = int(n_hidden_1 / 8)
n_hidden_3 = int(n_hidden_2 / 8)
'''

n_hidden_1 = 64
n_hidden_2 = 8
n_hidden_3 = 1

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="encoder_h1"),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="encoder_h2"),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name="encoder_h3"),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2]), name="decoder_h1"),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name="deocder_h2"),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name="decoder_h3")
}

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1]), name="encoder_b1"),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2]), name="encoder_b2"),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3]), name="encoder_b3"),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_2]), name="decoder_b1"),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_1]), name="decoder_b2"),
    'decoder_b3': tf.Variable(tf.zeros([n_input]), name="decoder_b3")
}

def encoder(x):
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
  layer_1 = tf.nn.dropout(layer_1, dropout_prob)
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
  return layer_3

def decoder(x):
  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
  return layer_3

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X
y_orig = Y

delta = y_true - y_pred
delta_orig = Y - Z

cost = tf.reduce_mean(tf.pow(delta, 2))
cost_orig = tf.reduce_mean(tf.pow(delta_orig, 2))

# If learning rate decay is implemented
'''
global_step = tf.Variable(0, trainable=False)
orig_learning_rate = 0.005
learning_rate = tf.train.exponential_decay(orig_learning_rate, global_step, 75000, 0.56)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
'''

# If learning rate decay is not implemented
learning_rate = 0.0024
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

error_range = [1.0 for x in range(data_num_size)]

if mod_min < 0:
  for i in range(index + 1):
    modifications[i] += mod_min

modifications_op = []
index_in_current_strides = 0
index = -1
for i in range(data_num_size):
  if (index == -1) or (index_in_current_strides == (strides[index] - 1)):
    index += 1
    index_in_current_strides = 0
    modifications_op.append(modifications[index])
  else:
    index_in_current_strides += 1
    modifications_op.append(modifications[index])

modifications = modifications_op

init = tf.global_variables_initializer()
sess.run(init)

bar = progressbar.ProgressBar(maxval=training_epochs, widgets=["Training Progress: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', progressbar.ETA()]).start()

total_batch = int(data_num_size / (n_input * batch_size))
if(data_num_size % (n_input * batch_size) != 0):
    total_batch += 1

batch_xs = [[[0 for x in range(n_input)] for y in range(batch_size)] for z in range(total_batch)]
original_values = [[[0 for x in range(n_input)] for y in range(batch_size)] for z in range(total_batch)]

# Fill batch_xs and original_values
index = 0
for i in range(total_batch):
  for j in range(batch_size):
    for k in range(n_input):
      if index < data_num_size:
        batch_xs[i][j][k] = data_num[index]
        original_values[i][j][k] = original_data_num[index]
        index += 1
      else:
        break

for epoch in range(training_epochs):
  bar.update(epoch + 1)
  if epoch == (training_epochs - 1):
    print("\nFinal Epoch")
  
  for i in range(total_batch):
    # Run the optimizer for the current batch_xs
    temp1 = time.time()
    _ = sess.run(optimizer, {X: batch_xs[i], dropout_prob: 0.75})
    temp2 = time.time()
    optimizer_sum += temp2 - temp1

    if epoch == (training_epochs - 1):
      p = sess.run(y_pred, {X: batch_xs[i], dropout_prob: 0.75})
      
      # For each predicted value, undo the modification that had been done on the original value of that prediction
      for r in range(np.size(p, 0)):
        for s in range(np.size(p, 1)):
          if ((i * batch_size * n_input) + (r * np.size(p, 1)) + s) < data_num_size:
            p[r][s] = p[r][s] / (10 ** modifications[((i * batch_size * n_input) + (r * np.size(p, 1)) + s)])
  
      c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs[i], Y: original_values[i], Z: p})
      print("Batch ", i, "- Cost: ", "{:.9f}".format(c), end='\t')

      temp1 = time.time()
      for a in range(batch_size):
        for b in range(n_input):
          current_index = i * batch_size * n_input + a * n_input + b
          try:
            if t[a][b] != 0:
              error_range[current_index] = (abs(d[a][b]) / t[a][b])
              error_sum += error_range[current_index]
              if(error_range[current_index] < 0.0001):
                error_A += 1
              elif(error_range[current_index] < 0.001):
                error_B += 1
              elif(error_range[current_index] < 0.01):
                error_C += 1
              elif(error_range[current_index] < 0.1):
                error_D += 1
              elif(error_range[current_index] < 1):
                error_E += 1
              else:
                error_F += 1

            # If the true value is 0
            else:
              error_range[current_index] = 0
              error_0 += 1
            
          except:
            break

      temp2 = time.time()
      error_organizing_sum += temp2 - temp1

  if epoch == (training_epochs - 1):
    print("\nFor the whole data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num_size), (error_A / data_num_size), (error_B / data_num_size), (error_C / data_num_size), (error_D / data_num_size), (error_E / data_num_size), (error_F / data_num_size), (error_sum / data_num_size)))
    
    # Reset the values of the error variables
    error_0 = 0    # if error = 0
    error_A = 0    # if 0 < error < 0.0001 (0.01%)
    error_B = 0    # if 0.0001 <= error < 0.001 (0.1%)
    error_C = 0    # if 0.001 <= error < 0.01 (1%)
    error_D = 0    # if 0.01 <= error < 0.1 (10%)
    error_E = 0    # if 0.1 <= error < 1 (100%)
    error_F = 0    # if 1 <= error
    error_sum = 0

training_end = time.time()
print("\nTraining complete!")
print("Total Training Time: %f seconds" % (training_end - training_start))
print("---------------------")
print("Normalize Data Method: %f seconds" % (normalize_data_method_end - normalize_data_method_start))
print("Optimizing Time: %f seconds" % optimizer_sum)
print("Error Organizing Time: %f seconds" % error_organizing_sum)


# Compression (testing)
print("\n\nCOMPRESSING")
print("Please choose a file for compression:")
f = files.upload()

name = ""

for f_i in f.keys():
  name = f_i

compression_start = time.time()

print("The name of the file for compression is:", name)

data_num = get_data(name)
original_data_num = np.copy(data_num)  # Copy data_num (by value, not reference) and store the copy in original_data_num
data_num, data_num_size, modifications, strides, mod_min, index = normalize_data(data_num)

if mod_min < 0:
  modifications = [x + mod_min for x in modifications]

modifications_op = [0] * data_num_size
index_in_current_strides = 0
index = -1
for i in range(data_num_size):
  if (index == -1) or (index_in_current_strides == (strides[index] - 1)):
    index += 1
    index_in_current_strides = 0
    modifications_op[i] = modifications[index]
  else:
    index_in_current_strides += 1
    modifications_op[i] = modifications[index]

modifications = modifications_op

ppoints = array.array('d', [0.0] * data_num_size)
zpoints = array.array('d')

total_batch = int(data_num_size / (n_input * batch_size))
if(data_num_size % (n_input * batch_size) != 0):
    total_batch += 1

batch_xs = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros

max_z = int(data_num_size / (n_input / n_hidden_3))
if (data_num_size % (n_input / n_hidden_3)) != 0:
  max_z += 1

index = 0  # Tracks the index of data_num, the vector of numbers

for i in range(total_batch):
  temp_batch_size = batch_size
  if i == (total_batch - 1):
    temp_batch_size = (data_num_size // n_input) - (batch_size * i)
    if data_num_size % n_input != 0:
      temp_batch_size += 1
  
  # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
  try:
    for j in range(temp_batch_size):
      for k in range(n_input):
        batch_xs[j][k] = data_num[index]
        original_values[j][k] = original_data_num[index]
        index += 1
  except:
    pass
  
  z_temp = sess.run(encoder_op, {X: batch_xs, dropout_prob: 1.0})

  for j in range(np.size(z_temp, 0)):
    for k in range(np.size(z_temp, 1)):
      if (i * np.size(z_temp, 0) + j * np.size(z_temp, 1) + k) < max_z:
        zpoints.append(float(z_temp[j][k]))

  p = sess.run(y_pred, {X: batch_xs, dropout_prob: 1.0})
  
  # For each predicted value, undo the modification that had been done on the original value of that prediction
  current_index = i * batch_size * n_input
  for r in range(np.size(p, 0)):
    for s in range(np.size(p, 1)):
      try:
        p[r][s] = p[r][s] / (10 ** modifications[current_index])
        ppoints[current_index] = p[r][s]
        current_index += 1
      except:
        continue
  
  d, t = sess.run([delta_orig, y_orig], feed_dict={Y: original_values, Z: p, dropout_prob: 1.0})
  
  current_index = i * batch_size * n_input
  for a in range(temp_batch_size):
    for b in range(n_input):
      try:
        if t[a][b] != 0:
          error = (abs(ppoints[current_index] - t[a][b]) / t[a][b])
          error_sum += error
          if(error < 0.0001):
            error_A += 1
          elif(error < 0.001):
            error_B += 1
          elif(error < 0.01):
            error_C += 1
          elif(error < 0.1):
            error_D += 1
          elif(error < 1):
            error_E += 1
          else:
            error_F += 1
        
        # If the true value is 0
        else:
          error_0 += 1
        
        current_index += 1
      
      except:
        break

print("For the testing data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num_size), (error_A / data_num_size), (error_B / data_num_size), (error_C / data_num_size), (error_D / data_num_size), (error_E / data_num_size), (error_F / data_num_size), (error_sum / data_num_size)))

compression_end = time.time()
print("\nCompression complete!")
print("Total Compressing Time: %f seconds" % (compression_end - compression_start))