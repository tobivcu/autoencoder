# Autoencoder that acts as a compressor and decompressor for HPC data by using machine learning
# Authors: Tong Liu and Shakeel Alibhai

import argparse
import array
import math
import numpy as np  
import os
import progressbar
import sys
import tensorflow as tf
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


'''
# For Multi-Threading
NUM_PARALLEL_EXEC_UNITS = 36
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "TRUE"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--training", help="Train the weights and biases by specifying a training file.")
parser.add_argument("-t", "--transfer", help="Use a new training file based on existing weights and biases to generate new weights and biases.")
parser.add_argument("-c", "--compress", help="Compress a file.")
parser.add_argument("-e", "--error", help="Set the error bound.")
parser.add_argument("-d", "--decompress", help="Decompress a file.")
parser.add_argument("-o", "--output_error", help="Print error information. Note: This argument currently only works with compression.", action="store_true")
args = parser.parse_args()

# If the user specified a file for compression and decompression, print a message stating that they cannot be done together and exit the program
if (args.compress != None) and (args.decompress != None):
    print("Compression and decompression cannot be done together!")
    sys.exit()

# If the user specifies an error bound but does not specify a file for compression, print a message and exit the program
if (args.compress == None) and (args.error != None):
    print("Error: Error bound specified but no compression file specified")
    sys.exit()

error_bound = 0

# If the user submitted an error-bound value as a parameter, convert the user's input to a float and save it in error_bound
if args.error != None:
    error_bound = float(args.error)

training_epochs = 10000
batch_size = 64
n_input = 4096

display = 5000  # In the training step, results are printed to a file every "display" number of epochs (as well as on the last epoch)

# Error definitions
error_0 = 0    # Number of elements where the error from the prediction is 0
error_A = 0    # Number of elements where 0 < error < 0.0001    (error between 0% and 0.01%)
error_B = 0    # Number of elements where 0 <= error < 0.001    (error between 0.01% and 0.1%)
error_C = 0    # Number of elements where 0.001 <= error < 0.01 (error between 0.1% and 1%)
error_D = 0    # Number of elements where 0.01 <= error < 0.1   (error between 1% and 10%)
error_E = 0    # Number of elements where 0.1 <= error < 1      (error between 10% and 100%)
error_F = 0    # Number of elements where error >= 1            (error greater than or equal to 100%)
error_sum = 0

sess = tf.compat.v1.Session()

# There are two possible ways of specifying the size of each layer of the autoencoder: by dividing the size of the previous layer by a certain amount or by simply specifying the size of each layer
# In this version, the latter case is used, while the former case is commented out
'''
n_hidden_1 = int(n_input / 4)
n_hidden_2 = int(n_hidden_1 / 5)
n_hidden_3 = int(n_hidden_2 / 5)
'''

n_hidden_1 = 256
n_hidden_2 = 16
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
    print("Length of file: %d bytes" % len(data))

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
        record_bytes = tf.decode_raw(data_node, tf.float32)
        data_num = sess.run(record_bytes, {data_node: data})

    if args.training != None:
        data_num = data_num
        #data_num = data_num[:838860]

    if args.transfer != None:
        transfer_size = int(input("Please enter tne number of elements from the transfer dataset to use for training: "))
        data_num = data_num[:transfer_size]

    return data_num

# Ensures that all of the data is either 0 or [0.01, 0.1)
# Note: This autoencoder is not currently intended to work with input files with negative numbers
def normalize_data(data_num):
    data_num_size = len(data_num)	# The number of data points in the input file

    min_val = np.amin(data_num)
    max_val = np.amax(data_num)
    range_val = max_val - min_val

    data_num = [((x - min_val) / range_val) for x in data_num]
    data_num = [((0.09 * x) + 0.01) for x in data_num]

    return data_num, data_num_size, min_val, range_val

# Returns True if a file with the specified name exists and False otherwise
def checkFileExists(file_name):
    try:
        f = open(file_name, "x")
        f.close()
        os.remove(file_name)
        return False
    except FileExistsError:
        return True

# If the user specified a file for training
if args.training != None:
    training_start = time.time()
    last_check_sum = 0.0
    fill_bo_sum = 0.0
    optimizer_sum = 0.0
    p_sum = 0.0
    p_proper_sum = 0.0
    cdt_sum = 0.0
    error_organizing_sum = 0.0
    bo_init_sum = 0.0
    print("\nTRAINING THE AUTOENCODER")
    print("---------------")
    file_name = args.training
    print("Training file: %s" % file_name)

    # Check to ensure that the training file exists. If it does not exist, print a message and exit the program.
    file_exists = checkFileExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()

    data_num = get_data(file_name)

    # TensorFlow Placeholders
    X = tf.placeholder("float", [None, None])	# Stores the normalized values from the input dataset
    Y = tf.placeholder("float", [None, n_input])	# Stores the original (non-normalized) values from the input dataset
    Z = tf.placeholder("float", [None, n_input])	# Stores the predicted values
    dropout_prob = tf.placeholder("float")	# Stores the probability of dropout during training

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

    # Open a file to write the error values to
    # This file has the same name as the training file with ".error" at the end
    error_log = open(file_name + ".error", 'w')

    # If learning rate decay is implemented
    '''
    global_step = tf.Variable(0, trainable=False)
    orig_learning_rate = 0.0025
    learning_rate = tf.train.exponential_decay(orig_learning_rate, global_step, 50000, 0.78)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    '''

    # If learning rate decay is not implemented
    learning_rate = 0.0024
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    normalize_data_method_start = time.time()
    original_data_num = np.copy(data_num)  # data_num will likely be modified, so make a copy of it and store it in original_data_num
    data_num, data_num_size, min_val, range_val = normalize_data(data_num)
    normalize_data_method_end = time.time()

    error_range = [1.0 for x in range(data_num_size)]

    # Initialize the weights and biases
    init = tf.global_variables_initializer()
    sess.run(init)

    # Start the progress bar
    bar = progressbar.ProgressBar(maxval=training_epochs, widgets=["Training Progress: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', progressbar.ETA()]).start()

    # Calculate the value of total_batch (the number of times the optimizer will be run every epoch)
    total_batch = int(data_num_size / (n_input * batch_size))

    # If the above division has a remainder, then increment the value of total_batch, as the optimizer will need to be run one more time to account for those remaining values
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1

    # total_batch is the number of total batches

    batch_xs = [[[0 for x in range(n_input)] for y in range(batch_size)] for z in range(total_batch)]
    original_values = [[[0 for x in range(n_input)] for y in range(batch_size)] for z in range(total_batch)]

    # Fill batch_xs and original_values
    temp1 = time.time()
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
    temp2 = time.time()
    fill_bo_sum = temp2 - temp1
 
    for epoch in range(training_epochs):
        bar.update(epoch + 1)     # Update the progress bar

        if(epoch % display == 0):
            print("Epoch %4d" % epoch, end='\t', file=error_log)

        index = 0  # Tracks the index of data_num, the vector of numbers

        for i in range(total_batch):
            # Run the optimizer for the current batch_xs (the weights and biases will be updated when this happens)
            temp1 = time.time()
            _ = sess.run(optimizer, {X: batch_xs[i], dropout_prob: 0.75})
            temp2 = time.time()
            optimizer_sum += (temp2 - temp1)

            # If the current epoch is one that should be printed to the file (this happens every "display" number of epochs and on the last epoch)
            if epoch % display == 0 or epoch == (training_epochs - 1):
                # Using the most recently updated weights and biases, send the current batch_xs through the encoder and decoder; the predicted values from the decoder will be stored in p
                temp1 = time.time()
                p = sess.run(y_pred, {X: batch_xs[i], dropout_prob: 0.75})
                temp2 = time.time()
                p_sum += temp2 - temp1

		# Using the normalized values, the original (non-normalized) values, and the predicted values, get the cost and delta values for each element
		# Save the costs, deltas, and original values in c, d, and t, respectively
                temp1 = time.time() 
                c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs[i], Y: original_values[i], Z: p})
                temp2 = time.time()
                cdt_sum += temp2 - temp1

                print("Batch", i, "- Cost: ", "{:.9f}".format(c), end='\t', file=error_log)
		
                temp1 = time.time()
                for a in range(batch_size):
                    for b in range(n_input):
                        current_index = i * batch_size * n_input + a * n_input + b
                        try:	# Ensure that the index is not out-of-range
                            # If the true value is not 0
                            if t[a][b] != 0:
                                new_x = (p[a][b] - 0.01) / 0.09
                                new_x = (new_x * range_val) + min_val
                                d[a][b] = t[a][b] - new_x

                                error_range[current_index] = (abs(d[a][b] / t[a][b]))

                                print("og: %.8f\tinput: %.8f\tpred: %.8f\tnew_x: %.8f" % (original_data_num[current_index], data_num[current_index], p[a][b], new_x), file=error_log)

                                # Add the current error value to error_sum, the sum of error values
                                error_sum += error_range[current_index]

                                # Increment the appropriate category of error values
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

        if epoch % display == 0 or epoch == (training_epochs - 1):
            print("For the whole data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num_size), (error_A / data_num_size), (error_B / data_num_size), (error_C / data_num_size), (error_D / data_num_size), (error_E / data_num_size), (error_F / data_num_size), (error_sum / data_num_size)), file=error_log)

            # Reset the values of the error variables
            error_0 = 0    # if error = 0
            error_A = 0    # if 0 < error < 0.0001(0.01%)
            error_B = 0    # if 0.0001 <= error < 0.001(0.1%)
            error_C = 0    # if 0.001 <= error < 0.01(1%)
            error_D = 0    # if 0.01 <= error < 0.1(10%)
            error_E = 0    # if 0.1 <= error < 1(100%)
            error_F = 0    # if 1 <= error
            error_sum = 0

    error_log.close()

    # Save the weight matrices and bias vectors
    save_start = time.time()
    saver = tf.train.Saver({
        "encoder_h1": weights['encoder_h1'],
        "encoder_h2": weights['encoder_h2'],
        "encoder_h3": weights['encoder_h3'],
        "decoder_h1": weights['decoder_h1'],
        "decoder_h2": weights['decoder_h2'],
        "decoder_h3": weights['decoder_h3'],
        "encoder_b1": biases['encoder_b1'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b1": biases['decoder_b1'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']
    })
    save_path = saver.save(sess, "./wb.ckpt", write_meta_graph=False)
    save_end = time.time()

    print("\n\nTraining complete!")
    print("Weight matrices and bias vectors stored in file: %s" % save_path)
    training_end = time.time()
    print("Training time: %f seconds" % (training_end - training_start))
    print("Normalize Data Method: %f seconds" % (normalize_data_method_end - normalize_data_method_start))
    print("Filling batch_xs and original_values: %f seconds" % fill_bo_sum)
    print("Optimizing Time: %f seconds" % optimizer_sum)
    print("p_time: %f seconds" % p_sum)
    print("p_proper_time: %f seconds" % p_proper_sum)
    print("cdt_time: %f seconds" % cdt_sum)
    print("Error Organizing Time: %f seconds" % error_organizing_sum)
    print("Saving Time: %f seconds" % (save_end - save_start))

# If the user specified a file for transfer learning
if args.transfer != None:
    training_start = time.time()
    last_check_sum = 0.0
    fill_bo_sum = 0.0
    optimizer_sum = 0.0
    p_sum = 0.0
    p_proper_sum = 0.0
    cdt_sum = 0.0
    error_organizing_sum = 0.0
    bo_init_sum = 0.0
    print("\nTRAINING THE AUTOENCODER (TRANSFER LEARNING)")
    print("---------------")
    file_name = args.transfer
    print("Training file: %s" % file_name)

    # Check to ensure that the training file exists. If it does not exist, print a message and exit the program.
    file_exists_start = time.time()
    file_exists = checkFileExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()
    file_exists_end = time.time()

    get_data_start = time.time()
    data_num = get_data(file_name)
    get_data_end = time.time()

    # TensorFlow Placeholders
    X = tf.placeholder("float", [None, None])	# Stores the normalized values from the input dataset
    Y = tf.placeholder("float", [None, n_input])	# Stores the original (non-normalized) values from the input dataset
    Z = tf.placeholder("float", [None, n_input])	# Stores the predicted values
    dropout_prob = tf.placeholder("float")	# Stores the probability of dropout during training

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
            'encoder_b1': tf.get_variable("encoder_b1", shape=[n_hidden_1]),
            'encoder_b2': tf.get_variable("encoder_b2", shape=[n_hidden_2]),
            'encoder_b3': tf.get_variable("encoder_b3", shape=[n_hidden_3]),
            'decoder_b1': tf.get_variable("decoder_b1", shape=[n_hidden_2]),
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
        "encoder_b1": biases['encoder_b1'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b1": biases['decoder_b1'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']})

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

    # Open a file to write the error values to
    # This file has the same name as the training file with ".error" at the end
    error_log = open(file_name + ".error", 'w')

    # If learning rate decay is implemented
    global_step = tf.Variable(0, trainable=False)
    orig_learning_rate = 0.0025
    learning_rate = tf.train.exponential_decay(orig_learning_rate, global_step, 50000, 0.78)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    sess = tf.compat.v1.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    # If learning rate decay is not implemented
    '''
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    '''
    
    normalize_data_method_start = time.time()
    original_data_num = np.copy(data_num)  # data_num will likely be modified, so make a copy of it and store it in original_data_num
    data_num, data_num_size, modifications, strides, mod_min, index = normalize_data(data_num)
    normalize_data_method_end = time.time()

    error_range = [1.0 for x in range(data_num_size)]

    setup_modifications_start = time.time()
    # If mod_min (the lowest value of modifications) was negative, then the absolute value of that number was added to every value of modifications in the normalize_data function, thus making every value of modifications either 0 or positive. Now we add mod_min back to every number to return to the original values. (Since mod_min is negative, we are essentially subtracting the value that we previously added.)
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
    setup_modifications_end = time.time()

    # Initialize the weights and biases
    wb_init_start = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    wb_init_end = time.time()

    # Multiply the existing weights and biases by a constant
    multiply_existing_we1 = tf.multiply(weights['encoder_h1'], 0.7)
    multiply_existing_we2 = tf.multiply(weights['encoder_h2'], 0.7)
    multiply_existing_we3 = tf.multiply(weights['encoder_h3'], 0.7)
    multiply_existing_wd1 = tf.multiply(weights['decoder_h1'], 0.7)
    multiply_existing_wd2 = tf.multiply(weights['decoder_h2'], 0.7)
    multiply_existing_wd3 = tf.multiply(weights['decoder_h3'], 0.7)
    multiply_existing_be1 = tf.multiply(biases['encoder_b1'], 0.7)
    multiply_existing_be2 = tf.multiply(biases['encoder_b2'], 0.7)
    multiply_existing_be3 = tf.multiply(biases['encoder_b3'], 0.7)
    multiply_existing_bd1 = tf.multiply(biases['decoder_b1'], 0.7)
    multiply_existing_bd2 = tf.multiply(biases['decoder_b2'], 0.7)
    multiply_existing_bd3 = tf.multiply(biases['decoder_b3'], 0.7)

    sess.run(multiply_existing_we1)
    sess.run(multiply_existing_we2)
    sess.run(multiply_existing_we3)
    sess.run(multiply_existing_wd1)
    sess.run(multiply_existing_wd2)
    sess.run(multiply_existing_wd3)
    sess.run(multiply_existing_be1)
    sess.run(multiply_existing_be2)
    sess.run(multiply_existing_be3)
    sess.run(multiply_existing_bd1)
    sess.run(multiply_existing_bd2)
    sess.run(multiply_existing_bd3)

    # Start the progress bar
    bar = progressbar.ProgressBar(maxval=training_epochs, widgets=["Training Progress: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(),' ', progressbar.ETA()]).start()

    # Calculate the value of total_batch (the number of times the optimizer will be run every epoch)
    total_batch = int(data_num_size / (n_input * batch_size))

    # If the above division has a remainder, then increment the value of total_batch, as the optimizer will need to be run one more time to account for those remaining values
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1
 
    for epoch in range(training_epochs):
        bar.update(epoch + 1)     # Update the progress bar
        temp1 = time.time()
        batch_xs = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
        original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros
        temp2 = time.time()
        bo_init_sum += temp2 - temp1

        if(epoch % display == 0):
            print("Epoch %4d" % epoch, end='\t', file=error_log)

        index = 0  # Tracks the index of data_num, the vector of numbers

        for i in range(total_batch):
            # If this is the last total_batch, then it may not be completely filled, as the final total_batch may contain the remaining values after division is rounded down
            temp1 = time.time()
            temp_batch_size = batch_size
            if i == (total_batch - 1):
                temp_batch_size = int(data_num_size / n_input) - (batch_size * i)
                if data_num_size % n_input != 0:
                    temp_batch_size += 1
            temp2 = time.time()
            last_check_sum += temp2 - temp1

            # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
            temp1 = time.time()
            try:
                for j in range(batch_size):
                    for k in range(n_input):
                        #if index < data_num_size:	# Check to ensure that the index is not out-of-range
                            batch_xs[j][k] = data_num[index]
                            original_values[j][k] = original_data_num[index]
                            index += 1
            except:
                print("", end='')
            temp2 = time.time()
            fill_bo_sum += temp2 - temp1

            # Run the optimizer for the current batch_xs (the weights and biases will be updated when this happens)
            temp1 = time.time()
            _ = sess.run(optimizer, {X: batch_xs, dropout_prob: 0.75})
            temp2 = time.time()
            optimizer_sum += (temp2 - temp1)

            # If the current epoch is one that should be printed to the file (this happens every "display" number of epochs and on the last epoch)
            if epoch % display == 0 or epoch == (training_epochs - 1):
                # Using the most recently updated weights and biases, send the current batch_xs through the encoder and decoder; the predicted values from the decoder will be stored in p
                temp1 = time.time()
                p = sess.run(y_pred, {X: batch_xs, dropout_prob: 0.75})
                temp2 = time.time()
                p_sum += temp2 - temp1

                # For each predicted value, undo the modification that had been done on the original value
                temp1 = time.time()
                for r in range(np.size(p, 0)):
                    for s in range(np.size(p, 1)):
                        if ((i * batch_size * n_input) + (r * np.size(p, 1)) + s) < data_num_size:	# Check to ensure that the index is not out-of-range
                            p[r][s] = p[r][s] / (10 ** modifications[((i * batch_size * n_input) + (r * np.size(p, 1)) + s)])
                temp2 = time.time()
                p_proper_sum += temp2 - temp1

		# Using the normalized values, the original (non-normalized) values, and the predicted values, get the cost and delta values for each element
		# Save the costs, deltas, and original values in c, d, and t, respectively
                temp1 = time.time() 
                c, d, t = sess.run([cost_orig, delta_orig, y_orig], feed_dict={X: batch_xs, Y: original_values, Z: p})
                temp2 = time.time()
                cdt_sum += temp2 - temp1

                print("Batch", i, "- Cost: ", "{:.9f}".format(c), end='\t', file=error_log)
		
                temp1 = time.time()
                for a in range(temp_batch_size):
                    for b in range(n_input):
                        current_index = i * batch_size * n_input + a * n_input + b
                        if current_index < data_num_size:	# Check to ensure that the index is not out-of-range
                            # If this is the last value in an input unit in the final epoch, print information about this value to the error file
                            if epoch == (training_epochs - 1) and b == (n_input - 1):
                                print("Epoch %4d\tInput Unit: %d\tt: %.8f\tarctan: %.8f\tp: %.8f\td: %.8f\tError: %.8f\tCost: %.16f" % (epoch, ((i * batch_size) + a), t[a][b], data_num[current_index],  p[a][b], d[a][b], (abs(d[a][b] / t[a][b])), c), file=error_log)

                            # Store the current error value in error_range, a list of error values
                            error_range[current_index] = (abs(d[a][b] / t[a][b]))

                            # Add the current error value to error_sum, the sum of error values
                            error_sum = error_sum + error_range[current_index]

                            # Increment the appropriate category of error values
                            if(error_range[current_index] == 0):
                                error_0 = error_0 + 1
                            elif(0 < error_range[current_index] < 0.0001):
                                error_A = error_A + 1
                            elif(0.0001 <=  error_range[current_index] < 0.001):
                                error_B = error_B + 1
                            elif(0.001 <= error_range[current_index] < 0.01):
                                error_C = error_C + 1
                            elif(0.01 <= error_range[current_index] < 0.1):
                                error_D = error_D + 1
                            elif(0.1 <= error_range[current_index] < 1):
                                error_E = error_E + 1
                            else:
                                error_F = error_F + 1
                temp2 = time.time()
                error_organizing_sum += temp2 - temp1

        if epoch % display == 0 or epoch == (training_epochs - 1):
            print("For the whole data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num_size), (error_A / data_num_size), (error_B / data_num_size), (error_C / data_num_size), (error_D / data_num_size), (error_E / data_num_size), (error_F / data_num_size), (error_sum / data_num_size)), file=error_log)

            # Reset the values of the error variables
            error_0 = 0    # if error = 0
            error_A = 0    # if 0 < error < 0.0001(0.01%)
            error_B = 0    # if 0.0001 <= error < 0.001(0.1%)
            error_C = 0    # if 0.001 <= error < 0.01(1%)
            error_D = 0    # if 0.01 <= error < 0.1(10%)
            error_E = 0    # if 0.1 <= error < 1(100%)
            error_F = 0    # if 1 <= error
            error_sum = 0

    error_log.close()

    # Save the weight matrices and bias vectors
    save_start = time.time()
    saver = tf.train.Saver({
        "encoder_h1": weights['encoder_h1'],
        "encoder_h2": weights['encoder_h2'],
        "encoder_h3": weights['encoder_h3'],
        "decoder_h1": weights['decoder_h1'],
        "decoder_h2": weights['decoder_h2'],
        "decoder_h3": weights['decoder_h3'],
        "encoder_b1": biases['encoder_b1'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b1": biases['decoder_b1'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']
    })
    save_path = saver.save(sess, "./wb.ckpt", write_meta_graph=False)
    save_end = time.time()

    print("\n\nTraining complete!")
    print("Weight matrices and bias vectors stored in file: %s" % save_path)
    training_end = time.time()
    print("Training time: %f seconds" % (training_end - training_start))
    print("Checking whether the training file exists: %f seconds" % (file_exists_end - file_exists_start))
    print("Get Data Method: %f seconds" % (get_data_end - get_data_start))
    print("Normalize Data Method: %f seconds" % (normalize_data_method_end - normalize_data_method_start))
    print("Setup Modifications: %f seconds" % (setup_modifications_end - setup_modifications_start))
    print("Initializing the Weights and Biases: %f seconds" % (wb_init_end - wb_init_start))
    print("Initializing batch_xs and original_values: %f seconds" % bo_init_sum)
    print("Checking whether it's the last total_batch: %f seconds" % last_check_sum)
    print("Filling batch_xs and original_values: %f seconds" % fill_bo_sum)
    print("Optimizing Time: %f seconds" % optimizer_sum)
    print("p_time: %f seconds" % p_sum)
    print("p_proper_time: %f seconds" % p_proper_sum)
    print("cdt_time: %f seconds" % cdt_sum)
    print("Error Organizing Time: %f seconds" % error_organizing_sum)
    print("Saving Time: %f seconds" % (save_end - save_start))

# If the user specified a file for compression
if args.compress != None:
    compression_start = time.time()
    last_check_sum = 0.0
    fill_bo_sum = 0.0
    get_z_layer_sum = 0.0
    p_sum = 0.0
    cdt_sum = 0.0
    bitmap_delta_error_processing_sum = 0.0
    error_organizing_sum = 0.0
    initialize_bitmap = 0.0
    restore_time = 0.0
    undo_modi = 0.0


    print("\nCOMPRESSING")
    print("---------------")
    file_name = args.compress
    print("File to compress: %s" % file_name)

    # Check to ensure that the file for compression exists. If it does not exist, print a message and exit the program.
    file_exists = checkFileExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()

    data_num = get_data(file_name)	# Get the data from the input file and store it in data_num
    normalize_data_start = time.time()
    original_data_num = np.copy(data_num)  # data_num will likely be modified, so make a copy of it and store it in original_data_num
    data_num, data_num_size, min_val, range_val = normalize_data(data_num)
    normalize_data_end = time.time() 

    # Write min_val to a file
    min_val_name = file_name + ".min_val"
    min_val_write = array.array("f", [min_val])
    with open(min_val_name, "wb") as f:
        f.write(bytes(min_val_write))

    # Write range_val to a file
    range_val_name = file_name + ".range_val"
    range_val_write = array.array("f", [range_val])
    with open(range_val_name, "wb") as f:
        f.write(bytes(range_val_write))


    temp1 = time.time()
    bitmap = Bitmap(data_num_size)
    temp2 = time.time()
    initialize_bitmap += temp2 - temp1


    # TensorFlow Placeholders
    X = tf.placeholder("float", [None, None])	# Stores the normalized values from the input dataset
    Y = tf.placeholder("float", [None, n_input])	# Stores the original (non-normalized) values from the input dataset
    Z = tf.placeholder("float", [None, n_input])	# Stores the predicted values

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
            'encoder_b1': tf.get_variable("encoder_b1", shape=[n_hidden_1]),
            'encoder_b2': tf.get_variable("encoder_b2", shape=[n_hidden_2]),
            'encoder_b3': tf.get_variable("encoder_b3", shape=[n_hidden_3]),
            'decoder_b1': tf.get_variable("decoder_b1", shape=[n_hidden_2]),
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
        "encoder_b1": biases['encoder_b1'],
        "encoder_b2": biases['encoder_b2'],
        "encoder_b3": biases['encoder_b3'],
        "decoder_b1": biases['decoder_b1'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']})

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
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

    temp1 = time.time()
    saver.restore(sess, "./wb.ckpt")
    temp2 = time.time()
    restore_time += temp2 - temp1

    zpoints = array.array('f')
    zname = file_name + ".z"
    zfile = open(zname, 'wb')

    ppoints = array.array('f', [0.0] * data_num_size)

    dipoints = array.array('I')
    diname = file_name + ".dindex"
    difile = open(diname, 'wb')

    dvpoints = array.array('f') # Stores the difference between the predicted value and the original value for elements whose prediction error is greater than the error bound
    dvname = file_name + ".dvalue"
    dvfile = open(dvname, 'wb')

    # Calculate the value of total_batch
    total_batch = int(data_num_size / (n_input * batch_size))

    # If the above division has a remainder, then increment the value of total_batch
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1

    batch_xs = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros
    original_values = [[0 for x in range(n_input)] for y in range(batch_size)]  # Initialize original_values to be filled with zeros
 
    max_z = int(data_num_size / (n_input / n_hidden_3))
    if (data_num_size % (n_input / n_hidden_3)) != 0:
        max_z += 1
 
    index = 0  # Tracks the index of data_num, the vector of numbers

    for i in range(total_batch):
        # If this is the last total_batch, then it may not be completely filled, as the final total_batch may contain the remaining values after division is rounded down
        temp_batch_size = batch_size
        if i == (total_batch - 1):
            temp_batch_size = (data_num_size // n_input) - (batch_size * i)
            if data_num_size % n_input != 0:
                temp_batch_size += 1


        # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
        temp1 = time.time()
        try:	# Ensure that the index is not out-of-range
            for j in range(temp_batch_size):
                for k in range(n_input):
                    batch_xs[j][k] = data_num[index]
                    original_values[j][k] = original_data_num[index]
                    index += 1
        except:
            pass
        temp2 = time.time()
        fill_bo_sum += temp2 - temp1

        # Run the encoder, which will return the z-layer of the neural network (this represents the compressed values of the input data)
        # Store the result in z_temp
        temp1 = time.time()
        z_temp = sess.run(encoder_op, {X: batch_xs})
        temp2 = time.time()
        get_z_layer_sum += temp2 - temp1

        for j in range(np.size(z_temp, 0)):
            for k in range(np.size(z_temp, 1)):
                if (i * np.size(z_temp, 0) + j * np.size(z_temp, 1) + k) < max_z:
                    zpoints.append(float(z_temp[j][k]))

        # Find what the predicted values would be (these are the values that the decoder would output, assuming that the weights and biases are the same (which they should be))
        temp1 = time.time()
        p = sess.run(y_pred, {X: batch_xs})
        temp2 = time.time()
        p_sum += temp2 - temp1


        temp1 = time.time()
        # For each predicted value, undo the modification that had been done on the original value of that prediction
        current_index = i * batch_size * n_input
        for r in range(np.size(p, 0)):
            for s in range(np.size(p, 1)):
                try:	# Ensure that the index is not out-of-range
                    p[r][s] = (p[r][s] - 0.01) / 0.09
                    p[r][s] = (p[r][s] * range_val) + min_val
                    ppoints[current_index] = p[r][s]
                    current_index += 1
                except:
                    continue
        temp2 = time.time()
        undo_modi += temp2 - temp1


        # Using the original values and the predicted values, get the delta value for each element
        # Save the deltas and original values in d and t, respectively
        temp1 = time.time()
        d, t = sess.run([delta_orig, y_orig], feed_dict={Y: original_values, Z: p})
        temp2 = time.time()
        cdt_sum += temp2 - temp1


        temp1 = time.time()
        # If the user specified an error-bound
        if args.error != None:
            current_index = i * batch_size * n_input
            for a in range(temp_batch_size):
                for b in range(n_input):
                    try:	# Ensure that the index is not out-of-range
		    	# If the current element's prediction error (the absolute value of the delta divided by the original value) is greater than the error bound
                        if (abs(d[a][b] / t[a][b])) >= error_bound:
			    # Mark the current index of the bitmap as an index where the prediction error is greater than the error bound
                            bitmap.set(current_index)

			    # Append the delta (the difference between the prediction error and the acutal value) to dvpoints
                            dvpoints.append(d[a][b])

                        current_index += 1

                    except:
                        break
        temp2 = time.time()
        bitmap_delta_error_processing_sum += temp2 - temp1


        temp1 = time.time()
        # If the user entered the parameter to output error information
        if args.output_error:
            error_log = open(file_name + ".error", 'w')
            current_index = i * batch_size * n_input
            for a in range(temp_batch_size):
                for b in range(n_input):
                    try:	# Ensure that the index is not out-of-range
                        # If the true value is not 0
                        if original_data_num[current_index] != 0:
                            error = abs((abs(ppoints[current_index]) - abs(original_data_num[current_index])) / original_data_num[current_index])

                            print("og: %.8f\tinput: %.8f\toutput: %.8f\terror: %.8f" % (original_data_num[current_index], data_num[current_index], ppoints[current_index], error), file=error_log)

                            # Add the current error value to error_sum, the sum of error values
                            error_sum += error

                            # Increment the appropriate category of error values
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

            print("For the testing data set, Error_0: %.8f\tError_A: %.8f\tError_B: %.8f\tError_C: %.8f\tError_D: %.8f\tError_E: %.8f\tError_F: %.8f\tError_mean: %.8f\t" % ((error_0 / data_num_size), (error_A / data_num_size), (error_B / data_num_size), (error_C / data_num_size), (error_D / data_num_size), (error_E / data_num_size), (error_F / data_num_size), (error_sum / data_num_size)), file=error_log)

            error_log.close()
        temp2 = time.time()
        error_organizing_sum += temp2 - temp1


    temp1 = time.time() 
    for item in bitmap.array:
        dipoints.append(item)

    # Add the delta values to one_dimen_p
    dvcounter = 0
    for a in range(data_num_size):
        if bitmap.test(a):
            ppoints[a] += dvpoints[dvcounter]
            dvcounter += 1
    
    dvarray = np.array(dvpoints, dtype='float16')
    temp2 = time.time()
    bitmap_delta_error_processing_sum += temp2 - temp1


    dvfile.write(bytes(dvarray))
    dipoints.tofile(difile)
    difile.close()
    dvfile.close()
    zpoints.tofile(zfile)
    zfile.close()

    # Get the size of the original file
    orig_file = open(file_name, "rb")
    orig_data = orig_file.read()
    orig_file.close()
    orig_len = len(orig_data)

    # Get the size of the .z (compressed) file
    comp_file = open(zname, "rb")
    comp_data = comp_file.read()
    comp_file.close()
    comp_len = len(comp_data)

    compression_end = time.time()
    print()
    print("Compression complete!")
    print("The compressed file %s has been successfully generated." % zname)
    print()
    print("Compression time: %f seconds" % (compression_end - compression_start))
    print("Compression throughput: %f MB/sec" % (data_num_size * 8 / (compression_end - compression_start) / 1048576))
    print("Normalize Data Method: %f seconds" % (normalize_data_end - normalize_data_start))
    #print("Setup Modifications: %f seconds" % (setup_modifications_end - setup_modifications_start))
    print("Filling batch_xs and original_values: %f seconds" % fill_bo_sum)
    print("Initialize the bitmap: %f seconds" % initialize_bitmap)
    print("Getting the z-layer: %f seconds" % get_z_layer_sum)
    print("Restore the network: %f seconds" % restore_time)
    print("Undo the modification: %f seconds" % undo_modi)
    print("p_time: %f seconds" % p_sum)
    print("cdt_time: %f seconds" % cdt_sum)
    print("Bitmap, Delta, and Error Processing: %f seconds" % bitmap_delta_error_processing_sum)
    print("Error Organizing Time: %f seconds" % error_organizing_sum)

# If the user specified a file for decompression
if args.decompress != None:
    start_time = time.time()
    print("\nDECOMPRESSING")
    print("---------------")
    file_name = args.decompress
    print("File to decompress: %s" % file_name)

    # Check to ensure that the file for decompression exists. If it does not exist, print a message and exit the program.
    file_exists = checkFileExists(file_name)
    if file_exists == False:
        print("Error: File does not exist.")
        sys.exit()

    # Store the data from the file to decompress in data_num
    data_num = get_data(file_name)

    # Get the size of data_num and store it in data_num_size
    size = tf.size(data_num)
    data_num_size = sess.run(size)

    Z_points = tf.placeholder("float", [None, None])

    # Set up the weight matrices (one for each layer of the decoder)
    weights = {
            'decoder_h1': tf.get_variable("decoder_h1", shape=[n_hidden_3, n_hidden_2]),
            'decoder_h2': tf.get_variable("decoder_h2", shape=[n_hidden_2, n_hidden_1]),
            'decoder_h3': tf.get_variable("decoder_h3", shape=[n_hidden_1, n_input])
    }

    # Set up the bias vectors (one for each layer of the decoder)
    biases = {
            'decoder_b1': tf.get_variable("decoder_b1", shape=[n_hidden_2]),
            'decoder_b2': tf.get_variable("decoder_b2", shape=[n_hidden_1]),
            'decoder_b3': tf.get_variable("decoder_b3", shape=[n_input])
    }

    # Get the weight matrices and bias vectors for the decoder from the training step
    saver = tf.train.Saver({
        "decoder_h1": weights['decoder_h1'],
        "decoder_h2": weights['decoder_h2'],
        "decoder_h3": weights['decoder_h3'],
        "decoder_b1": biases['decoder_b1'],
        "decoder_b2": biases['decoder_b2'],
        "decoder_b3": biases['decoder_b3']})

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
        return layer_3

    raw_file_name = file_name[:-2]  # Stores the name of the file for decompression without ".z" at the end

    decoder_op = decoder(Z_points)

    y_pred = decoder_op

    saver.restore(sess, "./wb.ckpt")

    # Get the saved information for the modifications list
    with open(raw_file_name + ".mod", "rb") as f:
        mod_temp = f.read()
    modifications = array.array('I', mod_temp)
    modifications = modifications.tolist()

    # Get the saved information for the strides list
    with open(raw_file_name + ".str", "rb") as f:
        str_temp = f.read()
    strides = array.array('L', str_temp)
    strides = strides.tolist()
    
    # Get the saved value of mod_min
    with open(raw_file_name + ".min", "rb") as f:
        min_temp = f.read()
    mod_min = array.array('b', min_temp)
    mod_min = mod_min.tolist()
    mod_min = mod_min[0]

    # Get the saved dindex information
    # This represents the indices of the input data where the predicted value's error is greater than the error bound
    with open(raw_file_name + ".dindex", "rb") as f:
        dindex = f.read()
    dindex = array.array('I', dindex)

    # Make a bitmap out of the dindex information
    bit_size = len(dindex) * 31
    bitmap_d = Bitmap(bit_size)
    arr_i = 0
    for item in dindex:
        bitmap_d.array[arr_i] = item
        arr_i += 1

    # Get the saved dvalue information
    # This represents the difference between the original values and the predicted values for all the data numbers whose predicted value's error is greater than the error bound
    # If a predicted value's error is greater than the error bound, the difference is saved in dvalue
    with open(raw_file_name + ".dvalue", "rb") as f:
        dvalue = np.fromfile(f, dtype=np.float16)
        dvalue = np.float32(dvalue)

    # If mod_min (the lowest value of modifications) was negative, then the absolute value of that number was added to every value of modifications in the normalize_data function so that every value of modifications would be either 0 or positive. Now we add mod_min back to every number to return to the original values. (Since mod_min is negative, we are essentially subtracting the value that we previously added.)
    if mod_min < 0:
        for i in range(len(modifications)): 
            modifications[i] += mod_min

    # Find the number of elements in the original, uncompressed data set; store this number in original_data_size
    original_data_size = 0
    for x in strides:
        original_data_size += x

    # Use the modifications and strides lists to make a new list, modifications_op, that has the length of the data_num from the compression step (the number of elements in the uncompressed file) and stores the modification value for that number such that the index numbers of modifications and the original data_num are aligned
    modifications_op = []
    index_in_current_strides = 0
    index = -1
    for i in range(original_data_size):
        # If this is the first pass of the for loop or if the current modification value is different from the previous modification value (ie. no longer included in the same stride)
        if (index == -1) or (index_in_current_strides == (strides[index] - 1)):
            index += 1
            index_in_current_strides = 0
            modifications_op.append(modifications[index])
        # If the modifications for the current index is the same as the modification for the previous index (ie. they are included in the same stride)
        else:
            index_in_current_strides += 1
            modifications_op.append(modifications[index])

    modifications = modifications_op
    
    print("Decompressing the .z file...")

    # Calculate the value of total_batch
    total_batch = int(data_num_size / (n_hidden_3 * batch_size))

    # If the above division has a remainder, then increment the value of total_batch
    if(data_num_size % (n_input * batch_size) != 0):
        total_batch += 1

    batch_xs = [[0 for x in range(n_hidden_3)] for y in range(batch_size)]  # Initialize batch_xs to be filled with zeros

    index = 0  # Tracks the index of data_num, the vector of numbers

    # Create an array to store the predicted values in, after they have been modified if necessary
    ppoints = array.array('f')

    for i in range(total_batch):
        # If this is the last total_batch, then it may not be completely filled, as the final total_batch may contain the remaining values after division is rounded down
        temp_batch_size = batch_size
        if i == (total_batch - 1):
            temp_batch_size = int(data_num_size / n_input) - (batch_size * i)
            if data_num_size % n_input != 0:
                temp_batch_size += 1

        # Put the next (batch_size * n_input) numbers from data_num into batch_xs and the next (batch_size * n_input) numbers from original_data_num into original_values
        for j in range(temp_batch_size):
            for k in range(n_hidden_3):
                if index < data_num_size:	# Check to ensure that the index is not out-of-range
                    batch_xs[j][k] = data_num[index]
                    index += 1
                
        # Run the deocder, which will return the output of the neural network (this represents the decompressed values of the input data)
        # Store the result in p
        p = sess.run(decoder_op, {Z_points: batch_xs})

    one_dimen_p = ppoints.tolist()

    # For all the values in the dataset, if the index is in dindex (represented by the bitmap), then add the dvalue back to the predicted value
    dvalue_index = 0
    for a in range(len(one_dimen_p)):
        if(bitmap_d.test(a) == 1):
            one_dimen_p[a] += dvalue[dvalue_index]
            dvalue_index += 1

    # Write the values from the decoder to a file
    k = open(file_name + ".txt", "w+")
    for x in one_dimen_p:
        k.write(str(x))
        k.write("\n")

    end_time = time.time()
    decompression_time = end_time - start_time

    print()
    print("Decompression complete!")
    print()
    print("The compressed file %s has been successfully decompressed to file %s." % (file_name, (file_name + ".txt")))
    print()
    print("Decompression Time: %f seconds" % decompression_time)
    print()
