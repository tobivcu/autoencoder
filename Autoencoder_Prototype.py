# Autoencoder that acts as a compressor and decompressor for HPC data by using machine learning
# Authors: Tong Liu and Shakeel Alibhai

import argparse
import array
import numpy as np  
import os
import progressbar
import sys
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--training", help="Train the weights and biases by specifying a training file.")
parser.add_argument("-c", "--compress", help="Compress a file.")
parser.add_argument("-e", "--error", help="Set the error bound.")
parser.add_argument("-d", "--decompress", help="Decompress a file.")
args = parser.parse_args()

# If the user specified a file for compression and decompression, print a message stating that they cannot be done together and exit the program
if (args.compress != None) and (args.decompress != None):
    print("Compression and decompression cannot be done together!")
    sys.exit()

error_bound = 0
if args.error != None:
    error_bound = float(args.error)

training_epochs = 25000
batch_size = 16
n_input = 256

display = 1000  # In the training step, results are printed to a file every "display" number of epochs (as well as on the last epoch)

# Error bound definitions
error_0 = 0    # Number of elements where the error from the prediction is 0
error_A = 0    # Number of elements where 0 < error < 0.0001    (error between 0% and 0.01%)
error_B = 0    # Number of elements where 0 <= error < 0.001    (error between 0.01% and 0.1%)
error_C = 0    # Number of elements where 0.001 <= error < 0.01 (error between 0.1% and 1%)
error_D = 0    # Number of elements where 0.01 <= error < 0.1   (error between 1% and 10%)
error_E = 0    # Number of elements where 0.1 <= error < 1      (error between 10% and 100%)
error_F = 0    # Number of elements where error >= 1            (error greater than or equal to 100%)
error_sum = 0

sess = tf.Session()

# There are two possible ways of specifying the size of each layer of the autoencoder: by dividing the size of the previous layer by a certain amount or by simply specifying the size of each layer
# In this version, the latter case is used, while the former case is commented out
'''
n_hidden_1 = int(n_input / 4)
n_hidden_2 = int(n_hidden_1 / 5)
n_hidden_3 = int(n_hidden_2 / 5)
'''

n_hidden_1 = 64
n_hidden_2 = 8
n_hidden_3 = 1

data_node = tf.placeholder(tf.string)

# Define the Bitmap class
class Bitmap(object):
    def __init__(self, max):
        self.size = self.calcElemIndex(max, True)
        self.array = [0 for i in range(self.size)]

    def calcElemIndex(self, num, up=False):
        # If up is true, then round up; otherwise, round down
        if up:
            return int((num + 31 - 1) // 31)    # Round up
        return num // 31

    def calcBitIndex(self, num):
        return num % 31

    def set(self, num):
        elemIndex = self.calcElemIndex(num)
        bitIndex = self.calcBitIndex(num)
        elem = self.array[elemIndex]
        self.array[elemIndex] = elem | (1 << bitIndex)

    def clean(self, i):
        elemIndex = self.calcElemIndex(i)
        bitIndex = self.calcBitIndex(i)
        elem = self.array[elemIndex]
        self.array[elemIndex] = elem & (~(1 << bitIndex))

    def test(self, i):
        elemIndex = self.calcElemIndex(i)
        bitIndex = self.calcBitIndex(i)
        if self.array[elemIndex] & (1 << bitIndex):
            return True
        return False 

# Convert the input data into a vector of numbers; the vector of numbers (data_num) is returned
def get_data(file_name):
    f = open(file_name, "rb")
    data = f.read()
    f.close()
    record_bytes = len(data)
    print("Length of file: %d bytes" % len(data))
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    # If the input file is a standard file (ie. not a binary file)
    if file_name[-4:] == ".txt":
        data_num = data.decode().split('\n')
        # If the input file is a standard file, there is a chance that the last line could simply be an empty line; if this is the case, then remove the empty line
        if(data_num[len(data_num) - 1] == ""):
            data_num.remove("")
        for i in range(len(data_num)):
            data_num[i] = float(data_num[i])
        data_num = np.array(data_num)

    # If the input file is a binary file
    else:
        record_bytes = tf.decode_raw(data_node, tf.float64)
        data_num = sess.run(record_bytes, {data_node: data})

    return data_num

# Ensures that all of the data is in a specific range (0 or [data_min, data_max])
def normalize_data(data_num):
    size = tf.size(data_num)
    data_num_size = sess.run(size)
    print("Number of elements in the file:", data_num_size)

    original_data_num = np.copy(data_num)  # data_num will likely be modified, so make a copy of it and store it in original_data_num
    modifications_all = []  # Stores the number of times that each number in the dataset was multiplied or divided by 10
    error_range = [1.0 for x in range(data_num_size)]

    # Specify the range to convert the data into, if it is not already in that range
    data_min = 0.01
    data_max = 0.1
        
    for i in range(data_num.size):
        modifications_value = 0     # Stores the number of times that the current value was multiplied (if positive) or divided (if negative) by 10

        # If the current value is 0, add 0 to modifications_all (indicating that this number was not multiplied or divided by 10) and go to the next element of the dataset
        if data_num[i] == 0:
            modifications_all.append(0)
            continue

        # This prototype does not support negative numbers
        '''if data_num[i] < 0:
            data_num[i] *= -1
            modifications[i] *= -1'''

        # Multiply the current value by 10 until it is equal to or greater than data_min; store the number of times this element has been multiplied by 10 in modifications_value
        while data_num[i] < data_min:
            data_num[i] *= 10.0
            modifications_value += 1

        # Divide the current value by 10 until it is equal to or less than data_max; store the number of times this element has been divided by 10 in modifications_value
        while data_num[i] > data_max:
            data_num[i] /= 10.0
            modifications_value -= 1
        
        # Add the number of times that the current value was multiplied or divided by 10 to modifications_all
        modifications_all.append(modifications_value)

    modifications = []
    strides = []
    index = -1
    last_num = 0.5  # Compare each element in modification_value to the previous element (stored in last_num); intialize last_num to 0.5 to avoid confusion at the beginning because each modification value should be an integer, so comparing to 0.5 should return False

    # Utilize the strides list to prevent the same number being stored next to itself in the modifications array
    for i in range(data_num_size):
        # If the modification value of the current index was the same as that of the previous index, then increment the strides that the previous index used by 1
        if modifications_all[i] == last_num:
            strides[index] += 1
        # If the modification value of the current index is different from that of the previous index, then add the value to modifications, set the strides at that value to be 1, update last_num, and increment the index (tracks the index of modifications and strides)
        else:
            modifications.append(modifications_all[i])
            strides.append(1)
            last_num = modifications_all[i]
            index += 1

    # Find the minimum value of modifications and, if it is negative, increment all the values in modifications by the absolute value of that number so that every value in modifications is 0 or positive
    mod_min = modifications[0]
    for i in range(index + 1):
        if modifications[i] < mod_min:
            mod_min = modifications[i]
    if mod_min < 0:
        for i in range(index + 1):
            modifications[i] += (mod_min * -1)

    return data_num, original_data_num, data_num_size, modifications, strides, error_range, mod_min, index

# Returns True if a file with the name of the output file already exists and False otherwise
def checkFileAlreadyExists(file_name):
    try:
        f = open(file_name, "x")
        f.close()
        return False
    except FileExistsError:
        return True

# If the user specified a file for training
if args.training != None:
    print("\nTRAINING THE AUTOENCODER")
    print("---------------")
    file_name = args.training
    print("Training file: %s" % file_name)

    # Check to ensure that the training file exists. If it does not exist, print a message and exit the program.
    file_exists = checkFileAlreadyExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()

    data_num = get_data(file_name)
        
    X = tf.placeholder("float", [None, None])
    Y = tf.placeholder("float", [None, n_input])
    Z = tf.placeholder("float", [None, n_input])

    dropout_prob = tf.placeholder("float")

    # Set up the weight matrices (one for each layer of the encoder and decoder); specify that their values should be initialized with values from the "random_normal" distribution when they are initialized
    weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="encoder_h1"),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="encoder_h2"),
            'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]), name="encoder_h3"),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2]), name="decoder_h1"),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]), name="decoder_h2"),
            'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input]), name="decoder_h3")
    }

    # Set up the bias vectors (one for each layer of the encoder and decoder); specify that their values should be initialized with zeros when they are initialized
    biases = {
            'encoder_b': tf.Variable(tf.zeros([n_hidden_1]), name="encoder_b"),
            'encoder_b2': tf.Variable(tf.zeros([n_hidden_2]), name="encoder_b2"),
            'encoder_b3': tf.Variable(tf.zeros([n_hidden_3]), name="encoder_b3"),
            'decoder_b': tf.Variable(tf.zeros([n_hidden_2]), name="decoder_b"),
            'decoder_b2': tf.Variable(tf.zeros([n_hidden_1]), name="decoder_b2"),
            'decoder_b3': tf.Variable(tf.zeros([n_input]), name="decoder_b3")
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b']))
        layer_1 = tf.nn.dropout(layer_1, dropout_prob)
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
        return layer_3

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b']))
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

    # Open a file to write the error values to
    # This file has the same name as the training file with ".error" at the end
    error_file_name = file_name + ".error"
    error_log1 = open(error_file_name, 'w')

    # If learning rate decay is implemented
    global_step = tf.Variable(0, trainable=False)
    orig_learning_rate = 0.0025
    learning_rate = tf.train.exponential_decay(orig_learning_rate, global_step, 50000, 0.78)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    # If learning rate decay is not implemented
    '''
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    '''
    
    data_num, original_data_num, data_num_size, modifications, strides, error_range, mod_min, index = normalize_data(data_num)

    # Undo the addition
    if mod_min < 0:
        for i in range(index + 1): 
            modifications[i] += mod_min

    # Use the modifications and strides lists to make a new list, modifications_op, that has the length of data_num and stores the modifications value for that number such that the index numbers of modifications and data_num are aligned
    modifications_op = []
    index_in_current_strides = 0
    index = -1
    for i in range(data_num_size):
        if (index == -1) or (index_in_current_strides == (strides[index] - 1)): # If this is the first pass of the for loop or if the current modifications value is different from the previous modifications value (ie. no longer included in the same stride)
            index += 1
            index_in_current_strides = 0
            modifications_op.append(modifications[index])
        else: # If the modification for the current index is the same as the modification for the previous index (ie. they are included in the same stride)
            index_in_current_strides += 1
            modifications_op.append(modifications[index])

    modifications = modifications_op

    # Initialize the weights and biases
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Start the progress bar
    bar = progressbar.ProgressBar(maxval=training_epochs, widgets=['Training Progress: ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', progressbar.ETA()], redirect_stderr=True).start()

    # Calculate the value of total_batch (the number of times the optimizer will be run every epoch)
    total_batch = int(data_num_size / (n_input * batch_size))

    # If the above division has a remainder, then increment the value of total_batch, as the optimizer will need to be run one more time to account for those remaining values
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1
 
    for epoch in range(training_epochs):
        bar.update(epoch + 1)     # Update the progress bar
        batch_xs = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
        original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros

        if(epoch % display == 0):
            print("Epoch %4d" % epoch, end='\t', file=error_log1)

        index = 0  # Tracks the index of data_num, the vector of numbers

        for i in range(total_batch):
            # If this is the last total_batch, then it will not be completely filled, as the final total_batch contains the remaining values after division is rounded down
            new_y = 0
            if i == (total_batch - 1):
                new_y = int(data_num_size / n_input) - (batch_size * i)
                if data_num_size % n_input != 0:
                    new_y += 1
            temp_batch_size = batch_size
            if i == (total_batch - 1):
                temp_batch_size = new_y

            # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
            for j in range(batch_size):
                for k in range(n_input):
                    if index < data_num_size:
                        batch_xs[j][k] = data_num[index]
                        original_values[j][k] = original_data_num[index]
                        index += 1

            # Run the optimizer for the current batch_xs (the weights and biases will be updated when this happens)
            _ = sess.run(optimizer, {X: batch_xs, dropout_prob: 0.75})

            # If the current epoch is one that should be printed to the file (this happens every "display" number of epochs and on the last epoch)
            if epoch % display == 0 or epoch == (training_epochs - 1):
                # Using the most recently updated weights and biases, send the current batch_xs through the encoder and decoder; the predicted values from the decoder will be stored in p
                p = sess.run(y_pred, {X: batch_xs, dropout_prob: 0.75})
                        
                # For each predicted value, undo the modification that had been done on the original value
                for r in range(np.size(p, 0)):
                    for s in range(np.size(p, 1)):
                        if ((i * batch_size * n_input) + (r * np.size(p, 1)) + s) < data_num_size:
                            p[r][s] = p[r][s] / (10 ** modifications[((i * batch_size * n_input) + (r * np.size(p, 1)) + s)])
                
                if epoch % display == 0 or epoch == (training_epochs - 1):
                    c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs, Y: original_values, Z: p})
                    print("Batch", i, "- Cost: ", "{:.9f}".format(c), end='\t', file=error_log1)
                          
                    temp_batch_size = batch_size
                    if i == (total_batch - 1):
                        temp_batch_size = new_y

                    for a in range(temp_batch_size):
                        for b in range(n_input):
                            if (i * batch_size * n_input + a * n_input + b) < data_num_size:
                                # If this is the last value in an input unit and the final epoch, print information about this value to the error file
                                if b == (n_input - 1) and epoch == (training_epochs - 1):
                                    print("Epoch %4d\tInput Unit: %d\tt: %.8f\tp: %.8f\td: %.8f\tError: %.8f\tCost: %.16f" % (epoch, ((i * batch_size) + a), t[a][b], p[a][b], d[a][b], (abs(d[a][b]) / t[a][b]), c), file=error_log1)
                                
                                # Store the current error value in error_range, a list of error values
                                error_range[i * batch_size * n_input + a * n_input + b] = (abs(d[a][b]) / t[a][b])

                                # Add the current error value to error_sum, the sum of error values
                                error_sum = error_sum + error_range[i * batch_size * n_input + a * n_input + b]

                                # Increment the appropriate category of error values
                                if(error_range[i * batch_size * n_input + a * n_input + b] == 0):
                                    error_0 = error_0 + 1
                                elif(0 < error_range[i * batch_size * n_input + a * n_input + b] < 0.0001):
                                    error_A = error_A + 1
                                elif(0.0001 <=  error_range[i * batch_size * n_input + a * n_input + b] < 0.001):
                                    error_B = error_B + 1
                                elif(0.001 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.01):
                                    error_C = error_C + 1
                                elif(0.01 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.1):
                                    error_D = error_D + 1
                                elif(0.1 <= error_range[i * batch_size * n_input + a * n_input + b] < 1):
                                    error_E = error_E + 1
                                else:
                                    error_F = error_F + 1
                                        
        if epoch % display == 0 or epoch == (training_epochs - 1):
            print("For the whole data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num.size),(error_A / data_num.size),(error_B / data_num.size),(error_C / data_num.size),(error_D / data_num.size),(error_E / data_num.size),(error_F / data_num.size),(error_sum / data_num.size)), file=error_log1)
                  
            # Reset the values of the error variables
            error_0 = 0    # if error = 0
            error_A = 0    # if 0 < error < 0.0001(0.01%)
            error_B = 0    # if 0.0001 <= error < 0.001(0.1%)
            error_C = 0    # if 0.001 <= error < 0.01(1%)
            error_D = 0    # if 0.01 <= error < 0.1(10%)
            error_E = 0    # if 0.1 <= error < 1(100%)
            error_F = 0    # if 1 <= error
            error_sum = 0

    error_log1.close()

    # Save the weight matrices and bias vectors
    saver = tf.train.Saver({
        "encoder_h1": weights['encoder_h1'],
        "encoder_h2": weights['encoder_h2'],
        "encoder_h3": weights['encoder_h3'],
        "decoder_h1": weights['decoder_h1'],
        "decoder_h2": weights['decoder_h2'],
        "decoder_h3": weights['decoder_h3'],
        "encoder_b": biases['encoder_b'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b": biases['decoder_b'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']
    })
    save_path = saver.save(sess,"./wb.ckpt")
    print("Training complete!")
    print("Weight matrices and bias vectors stored in file: %s" % save_path)

# If the user specified a file for compression
if args.compress != None:
    print("\nCOMPRESSING")
    print("---------------")
    start_time  = time.time()
    file_name = args.compress
    print("File to compress: %s" % file_name)

    # Check to ensure that the file for compression exists. If it does not exist, print a message and exit the program.
    file_exists = checkFileAlreadyExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()

    data_num = get_data(file_name)
    data_num, original_data_num, data_num_size, modifications, strides, error_range, mod_min, index = normalize_data(data_num)
       
    bitmap = Bitmap(data_num_size)

    X = tf.placeholder("float", [None, None])
    Y = tf.placeholder("float", [None, n_input])
    Z = tf.placeholder("float", [None, n_input])

    # Set up the weight matrices (one for each layer of the encoder and decoder)
    weights = {
            'encoder_h1': tf.get_variable("encoder_h1", shape=[n_input, n_hidden_1]),
            'encoder_h2': tf.get_variable("encoder_h2", shape=[n_hidden_1, n_hidden_2]),
            'encoder_h3': tf.get_variable("encoder_h3", shape=[n_hidden_2, n_hidden_3]),
            'decoder_h1': tf.get_variable("decoder_h1", shape=[n_hidden_3, n_hidden_2]),
            'decoder_h2': tf.get_variable("decoder_h2", shape=[n_hidden_2, n_hidden_1]),
            'decoder_h3': tf.get_variable("decoder_h3", shape=[n_hidden_1, n_input])
    }

    # Set up the bias vectors (one for each layer of the encoder and decoder)
    biases = {
            'encoder_b': tf.get_variable("encoder_b", shape=[n_hidden_1]),
            'encoder_b2': tf.get_variable("encoder_b2", shape=[n_hidden_2]),
            'encoder_b3': tf.get_variable("encoder_b3", shape=[n_hidden_3]),
            'decoder_b': tf.get_variable("decoder_b", shape=[n_hidden_2]),
            'decoder_b2': tf.get_variable("decoder_b2", shape=[n_hidden_1]),
            'decoder_b3': tf.get_variable("decoder_b3", shape=[n_input])
    }

    # Get the weights and biases from the training step
    saver = tf.train.Saver({
        "encoder_h1": weights['encoder_h1'],
        "encoder_h2": weights['encoder_h2'],
        "encoder_h3": weights['encoder_h3'],
        "decoder_h1": weights['decoder_h1'],
        "decoder_h2": weights['decoder_h2'],
        "decoder_h3": weights['decoder_h3'],
        "encoder_b": biases['encoder_b'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b": biases['decoder_b'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']})

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
        return layer_3

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b']))
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

    saver.restore(sess, "./wb.ckpt")

    modifications_name = file_name + ".mod"
    modifications_write = array.array("I", modifications)

    with open(modifications_name, "wb") as f:
            f.write(bytes(modifications_write))

    strides_name = file_name + ".str"
    strides_write = array.array("L", strides)

    with open(strides_name, "wb") as f:
            f.write(bytes(strides_write))

    mod_min_name = file_name + ".min"
    mod_min_write = array.array("b", [mod_min])

    with open(mod_min_name, "wb") as f:
            f.write(bytes(mod_min_write))

    with open(file_name + ".mod", "rb") as f:
            mod_temp = f.read()
    modifications = array.array('I', mod_temp)

    with open(file_name + ".str", "rb") as f:
            str_temp = f.read()
    strides = array.array('L', str_temp)

    with open(file_name + ".min", "rb") as f:
            min_temp = f.read()
    mod_min = array.array('b', min_temp)

    modifications = modifications.tolist()
    strides = strides.tolist()
    mod_min = mod_min.tolist()
    mod_min = mod_min[0]

    # Undo the addition
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

    error_file_name = file_name + ".error"
    error_log2 = open(error_file_name,'w')

    zpoints = array.array('d')
    zname = file_name + ".z"
    zfile = open(zname, 'wb')       

    ppoints = array.array('d')

    dipoints = array.array('I')
    diname = file_name + ".dindex"
    difile = open(diname, 'wb')

    dvpoints = array.array('f')
    dvname = file_name + ".dvalue"
    dvfile = open(dvname, 'wb')

    total_batch = int(data_num_size / (n_input * batch_size))
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1

    batch_xs = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
    original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros
          
    index = 0  # Tracks the index of data_num, the vector of numbers
    end_time = time.time()
    time_sum = end_time - start_time

    for i in range(total_batch):
        start_time2 = time.time()

        # If this is the last total_batch, then it will not be completely filled, as the final total_batch contains the remaining values after division is rounded down
        new_y = 0
        if i == (total_batch - 1):
            new_y = int(data_num_size / n_input) - (batch_size * i)
            if data_num_size % n_input != 0:
                new_y += 1
        temp_batch_size = batch_size
        if i == (total_batch - 1):
              temp_batch_size = new_y

        # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
        for j in range(temp_batch_size):
            for k in range(n_input):
                if index < data_num_size:
                    batch_xs[j][k] = data_num[index]
                    original_values[j][k] = original_data_num[index]
                    index += 1

        # Run the encoder, which will return the z-layer of the neural network (this represents the compressed values of the input data)
        # Store the result in z_temp
        z_temp = sess.run(encoder_op, {X: batch_xs})
            
        for j1 in range(np.size(z_temp, 0)):
            for k1 in range(np.size(z_temp, 1)):
                max_z = int(data_num_size / (n_input / n_hidden_3))
                if (data_num_size % (n_input / n_hidden_3)) != 0:
                    max_z += 1
                if (i * np.size(z_temp, 0) + j1 * np.size(z_temp, 1)+ k1) < max_z:
                    zpoints.append(float(z_temp[j1][k1]))

        end_time2 = time.time()
        time_sum += end_time2 - start_time2

        p = sess.run(y_pred, {X: batch_xs})
                
        # For each predicted value, undo the modification that had been done on the original value of that prediction
        for r in range(np.size(p, 0)):
            for s in range(np.size(p, 1)):
                if ((i * batch_size * n_input) + (r * np.size(p, 1)) + s) < data_num_size:
                    p[r][s] = p[r][s] / (10 ** modifications[((i * batch_size * n_input) + (r * np.size(p, 1)) + s)])
                    ppoints.append(float(p[r][s]))

        c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs, Y: original_values, Z: p})
        one_dimen_p = ppoints.tolist()

        for a in range(temp_batch_size):
            for b in range(n_input):
                if (i * batch_size * n_input + a * n_input + b) < data_num_size:
                    error_range[i * batch_size * n_input + a * n_input + b] = (abs(d[a][b])/t[a][b])
                    if args.error != None:
                        if error_range[i * batch_size * n_input + a * n_input + b] >= error_bound:
                            bitmap.set(i*batch_size*n_input + a*n_input + b)
                            dvpoints.append(d[a][b])

        dvcounter = 0
        for i1 in range(data_num_size):
            if bitmap.test(i1):
                one_dimen_p[i1] += dvpoints[dvcounter]
                dvcounter += 1

        for a in range(temp_batch_size):
            for b in range(n_input):
                if (i * batch_size * n_input + a * n_input + b) < data_num_size:
                    # If the true value is not 0
                    if t[a][b] != 0:  
                        error_range[i * batch_size * n_input + a * n_input + b] = (abs(one_dimen_p[i*batch_size*n_input + a*n_input + b] - t[a][b])/t[a][b])

                    # If the true value is 0
                    else:
                        error_range[i * batch_size * n_input + a * n_input + b] = 0

                    # Print information about this value to the error file
                    print("Testing Input Unit %d\tt: %.8f\tp: %.8f\td: %.8f\tError: %.8f\tCost: %.16f" % (((i * batch_size) + a), t[a][b], one_dimen_p[i*batch_size*n_input + a*n_input + b], one_dimen_p[i*batch_size*n_input + a*n_input + b] - t[a][b], error_range[i * batch_size * n_input + a * n_input + b], c), file=error_log2)

                    # Add the current error value to error_sum, the sum of error values
                    error_sum = error_sum + error_range[i * batch_size * n_input + a * n_input + b]

                    # Increment the appropriate category of error values
                    if(error_range[i * batch_size * n_input + a * n_input + b] == 0):
                        error_0 = error_0 + 1
                    elif(0 < error_range[i * batch_size * n_input + a * n_input + b] < 0.0001):
                        error_A = error_A + 1
                    elif(0.0001 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.001):
                        error_B = error_B + 1
                    elif(0.001 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.01):
                        error_C = error_C + 1
                    elif(0.01 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.1):
                        error_D = error_D + 1
                    elif(0.1 <= error_range[i * batch_size * n_input + a * n_input + b] < 1):
                        error_E = error_E + 1
                    else:
                        error_F = error_F + 1
          
    new_start = time.time()
 
    for item in bitmap.array:
        dipoints.append(item)

    dvarray = np.array(dvpoints, dtype='float16')
    dvfile.write(bytes(dvarray))
    dipoints.tofile(difile)
    difile.close()
    dvfile.close()
    zpoints.tofile(zfile)
    zfile.close()
    new_end = time.time()
    time_sum += new_end - new_start
    print("For the testing data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num.size), (error_A / data_num.size), (error_B / data_num.size), (error_C / data_num.size), (error_D / data_num.size), (error_E / data_num.size), (error_F / data_num.size), (error_sum / data_num.size)), file=error_log2)
    fo = open(file_name, "rb")
    datao = fo.read()
    fo.close()
    lengtho = len(datao)
    fc = open(zname,"rb")
    datac = fc.read()
    fc.close()
    lengthc = len(datac)
    print()
    print("Compression complete!")
    print()
    print("The compressed file %s has been successfully generated with a compression ratio of %f." % (zname, (lengtho / lengthc)))
    print("Compression error information is in the error logs.")
    print()
    print("Compression Time: %f seconds\nCompression Throughput: %f MB/s" % (time_sum, lengtho / (time_sum * 1024 * 1024)))
    print()

# If the user specified a file for decompression
if args.decompress != None:
        print("DECOMPRESSING\n----------\n")
        start_time  = time.time()
        file_name = args.decompress
        print("Testing file: %s" % file_name)

        file_exists = checkFileAlreadyExists(file_name)
        if file_exists == False:
                print("Error: File does not exist.")
                sys.exit()
        data_num = get_data(file_name)
        data_num, original_data_num, data_num_size, modifications, strides, error_range, mod_min, index = normalize_data(data_num)

        X = tf.placeholder("float", [None, None])
        Y = tf.placeholder("float", [None, n_input])
        Z = tf.placeholder("float", [None, n_input])
        Z_points = tf.placeholder("float", [None, None])

        weights = {
                'encoder_h1': tf.get_variable("encoder_h1", shape=[n_input, n_hidden_1]),
                'encoder_h2': tf.get_variable("encoder_h2", shape=[n_hidden_1, n_hidden_2]),
                'encoder_h3': tf.get_variable("encoder_h3", shape=[n_hidden_2, n_hidden_3]),
                'decoder_h1': tf.get_variable("decoder_h1", shape=[n_hidden_3, n_hidden_2]),
                'decoder_h2': tf.get_variable("decoder_h2", shape=[n_hidden_2, n_hidden_1]),
                'decoder_h3': tf.get_variable("decoder_h3", shape=[n_hidden_1, n_input])
        }

        biases = {
                'encoder_b': tf.get_variable("encoder_b", shape=[n_hidden_1]),
                'encoder_b2': tf.get_variable("encoder_b2", shape=[n_hidden_2]),
                'encoder_b3': tf.get_variable("encoder_b3", shape=[n_hidden_3]),
                'decoder_b': tf.get_variable("decoder_b", shape=[n_hidden_2]),
                'decoder_b2': tf.get_variable("decoder_b2", shape=[n_hidden_1]),
                'decoder_b3': tf.get_variable("decoder_b3", shape=[n_input])
        }

        saver = tf.train.Saver({"encoder_h1": weights['encoder_h1'],
                "encoder_h2": weights['encoder_h2'],
                "encoder_h3": weights['encoder_h3'],
                "decoder_h1": weights['decoder_h1'],
                "decoder_h2": weights['decoder_h2'],
                "decoder_h3": weights['decoder_h3'],
                "encoder_b": biases['encoder_b'],
                "encoder_b2": biases['encoder_b2'],
                "encoder_b3": biases['encoder_b3'],
                "decoder_b": biases['decoder_b'],
                "decoder_b2": biases['decoder_b2'],
                "decoder_b3": biases['decoder_b3']})

        def decoder(x):
          layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b']))
          layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
          layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
          return layer_3

        with open(args.decompress, 'rb') as f:
          zfile_temp = f.read()
        znumbers = array.array('d',zfile_temp)

        #encoder_op = encoder(X)
        decoder_op = decoder(Z_points)

        y_pred = decoder_op
        y_true = X
        y_orig = Y

        delta = y_true - y_pred
        delta_orig = Y - Z

        cost = tf.reduce_mean(tf.pow(delta, 2))
        cost_orig = tf.reduce_mean(tf.pow(delta_orig, 2))

        saver.restore(sess, "./wb.ckpt")

        modifications_name = file_name + ".mod"
        modifications_write = array.array("B", modifications)

        with open(modifications_name, "wb") as f:
                f.write(bytes(modifications_write))

        strides_name = file_name + ".str"
        strides_write = array.array("L", strides)

        with open(strides_name, "wb") as f:
                f.write(bytes(strides_write))

        mod_min_name = file_name + ".min"
        mod_min_write = array.array("b", [mod_min])

        with open(mod_min_name, "wb") as f:
                f.write(bytes(mod_min_write))


        with open(file_name+ ".mod", "rb") as f:
                mod_temp = f.read()
        modifications = array.array('B', mod_temp)

        with open(file_name+ ".str", "rb") as f:
                str_temp = f.read()
        strides = array.array('L', str_temp)

        with open(file_name+ ".min", "rb") as f:
                min_temp = f.read()
        mod_min = array.array('b', min_temp)

        modifications = modifications.tolist()
        strides = strides.tolist()
        mod_min = mod_min.tolist()
        mod_min = mod_min[0]

        # Undo the addition
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

        print("Decompressing the .z file...")

        error_file_name = file_name + ".error"
        error_log2 = open(error_file_name,'w')

        zpoints = array.array('d')
        zname = file_name + ".z"
        zfile = open(zname, 'wb')       

        ppoints = array.array('d')
        pname = file_name + ".d"
        pfile = open(pname, 'wb')

        dipoints = array.array('H')
        dipoints = dipoints.tolist()
        diname = file_name + ".dindex"
        difile = open(diname, 'wb')

        dvpoints = array.array('f')
        dvpoints = dvpoints.tolist()
        dvname = file_name + ".dvalue"
        dvfile = open(dvname, 'wb')

        if True:
          total_batch = int(data_num_size / (n_input * batch_size))
          if(data_num_size % (n_input * batch_size) != 0):
            total_batch += 1

          data_num = znumbers.tolist()
          data_num_size = len(data_num)

          batch_xs = [[0 for x in range(n_hidden_3)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
          #original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros
          
          index = 0  # Tracks the index of data_num, the vector of numbers
          end_time =time.time()
          time_sum = end_time - start_time

          for i in range(total_batch):
            start_time2 = time.time()
            new_y = 0
                # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
            if i == (total_batch - 1):
              new_y = int(data_num_size / n_input) - (batch_size * i)
              if data_num_size % n_input != 0:
                new_y += 1
              #batch_xs = [[0 for x in range(n_input)] for y in range(new_y)]
              #original_values = [[0 for x in range(n_input)] for y in range(new_y)]
            temp_batch_size = batch_size
            if i == (total_batch - 1):
              temp_batch_size = new_y
            for j in range(temp_batch_size):
              for k in range(n_hidden_3):
                if index < data_num_size:
                  batch_xs[j][k] = data_num[index]
                  #original_values[j][k] = original_data_num[index]
                  index += 1
                
            end_time2 = time.time()
            time_sum += end_time2 - start_time2
            p = sess.run(decoder_op, {Z_points: batch_xs})
                
            # For each predicted value, undo the modification that had been done on the original value of that prediction
            for r in range(np.size(p, 0)):
              for s in range(np.size(p, 1)):
                if ((i * batch_size * n_input) + (r * np.size(p, 1)) + s) < data_num_size:
                  p[r][s] = p[r][s] / (10 ** modifications[((i * batch_size * n_input) + (r * np.size(p, 1)) + s)])
                  ppoints.append(float(p[r][s]))

            with open(file_name + ".dindex", "rb") as f:
              dindex_temp = f.read()
              dindices = array.array('H', dindex_temp)
            dindices = dindices.tolist()

            with open(file_name + ".dvalue", "rb") as f:
              dvalue_temp = f.read()
              dvalues = array.array('f', dvalue_temp)
            dvalues = dvalues.tolist()


            #c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs, Y: original_values, Z: p})
            one_dimen_p = ppoints.tolist()
            # Print each input unit index, its t value, its p value, its d value, its relative error, and its cost
           # print()
            '''for a in range(temp_batch_size):
              for b in range(n_input):
                if (i * batch_size * n_input + a * n_input + b) < data_num_size:
                  error_range[i * batch_size * n_input + a * n_input + b] = (abs(d[a][b])/t[a][b])
                  if args.error != None:
                    if error_range[i * batch_size * n_input + a * n_input + b] >= error_bound:
                      dipoints.append(i*batch_size*n_input + a*n_input + b)
                      dvpoints.append(d[a][b])'''
            for a in range(len(dipoints)):
              ii = dipoints[a]
              one_dimen_p[ii] += dvpoints[a]

            temp_batch_size = batch_size
            if i == (total_batch - 1):
              temp_batch_size = new_y
            '''for a in range(temp_batch_size):
              for b in range(n_input):
                if(i * batch_size * n_input + a * n_input + b) < data_num_size:
                  print("Testing Input Unit %d\tt: %.8f\tp: %.8f\td: %.8f\tError: %.8f\tCost: %.16f" % (((i * batch_size) + a), t[a][b], one_dimen_p[i*batch_size*n_input + a*n_input + b], one_dimen_p[i*batch_size*n_input + a*n_input + b] - t[a][b], (abs(one_dimen_p[i*batch_size*n_input + a*n_input + b] - t[a][b])/t[a][b]), c), file=error_log2)'''
                
            #delta_index = []
            #delta_value = []


            '''
            dipoints.tofile(difile)
            dvpoints.tofile(dvfile)

            with open(file_name + ".dindex", "rb") as f:
              dindex_temp = f.read()
              dindices = array.array('L', dindex_temp)
            dindices = dindices.tolist()

            with open(file_name + ".dvalue", "rb") as f:
              dvalue_temp = f.read()
              dvalues = array.array('f', dvalue_temp)
            dvalues = dvalues.tolist()
            '''

                

            # Print the batch index and its cost
           # print("Batch", i, "- Cost: ", "{:.9f}".format(c), end = '\t')
            '''for a in range(temp_batch_size):
              for b in range(n_input):
                if (i * batch_size * n_input + a * n_input + b) < data_num_size:
                  error_range[i * batch_size * n_input + a * n_input + b] = (abs(one_dimen_p[i*batch_size*n_input + a*n_input + b] - t[a][b])/t[a][b])
                  error_sum = error_sum + error_range[i * batch_size * n_input + a * n_input + b]
                  if(error_range[i * batch_size * n_input + a * n_input + b] == 0):
                    error_0 = error_0 + 1
                  if(0 < error_range[i * batch_size * n_input + a * n_input + b] < 0.0001):
                    error_A = error_A + 1
                  if(0.0001 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.001):
                    error_B = error_B + 1
                  if(0.001 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.01):
                    error_C = error_C + 1
                  if(0.01 <= error_range[i * batch_size * n_input + a * n_input + b] < 0.1):
                    error_D = error_D + 1
                  if(0.1 <= error_range[i * batch_size * n_input + a * n_input + b] < 1):
                    error_E = error_E + 1
                  if(1 <= error_range[i * batch_size * n_input + a * n_input + b] ):
                    error_F = error_F + 1'''
          new_start = time.time()
          dvfile.close()
          zfile.close()
          new_end = time.time()
          time_sum += new_end - new_start
          ppoints.tofile(pfile)
          pfile.close()
          fo = open(file_name, "rb")
          datao = fo.read()
          fo.close()
          lengtho = len(datao)
          fc = open(zname,"rb")
          datac = fc.read()
          fc.close()
          lengthc = len(datac)
          print()
          print("Decompression complete!\n")
