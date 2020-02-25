# Autoencoder for Compression and Decompression
Authors: Tong Liu (tongliu@temple.edu) and Shakeel Alibhai (shakeel.alibhai@temple.edu)

## Overview
This is an autoencoder that uses machine learning to compress and decompress files. It first takes a training file that is used to train the autoencoder, thus generating weight matrices and bias vectors. It will then save the weight matrices and bias vectors so that they can be used for compression and/or decompression.

## Usage
### Setting the Parameters
There are several parameters that can be modified in the program. These include, but are not necessarily limited to, the following:

* Number of Epochs
* Learning Rate/Learning Rate Decay
* Batch Size
* *n_input*

These parameters can be modified by changing the code of the program. The values that we recommend for these variables are the ones that are already set; however, these may not be optimal for all datasets.

### Specifying a Compression Ratio
Currently, the only way to specify a theoretical compression ratio is to modify the code. The compression ratio is equal to the product of `n_input`, `n_hidden1`, `n_hidden2`, and `n_hidden3`. Note, however, that this is only a theoretical compression ratio; the actual compression ratio may be lower than the theoretical compression ratio. This may occur for several reasons, such as the value of the error bound, the range of the input data and how well the autoencoder is optimized to predict the data.

### Running the Autoencoder
#### Arguments
* `-r` for training file
* `-c` for compression file
* `-d` for decompression file
* `-e` for error-bound
* `-t` for transfer learning
* `-o` for compression error information (only in compression step)

#### Examples
* Example with "train.txt" as a training file and "data.txt" as a file to compress: `python3 Autoencoder_Prototype.py -r train.txt -c data.txt`
* Example with "test.bin" as a file to compress and an error bound of 10%: `python3 Autoencoder_Prototype.py -c test.bin -e 0.1` (Note: In order to compress, weights and biases must have been previously generated. The weights and biases should be in the same folder as the current working directory and autoencoder code.)
* Example with "p.z" as a file to decompress: `python3 Autoencoder_Prototype.py -d p.z` (Note: In this case, "p.z" must be the output file from the compression step of the autoencoder.)

#### Notes
* Compression and decompression cannot be done together.
* If an error bound is specified, then a file for compression must be specified as well. (However, it is possible to specify a file for compression without specifying an error bound.)
* The compression step will produce a .z file, as well as several other files. All the files should be kept there for decompression; however, when entering the command to start the autoencoder for decompression, only the .z file needs to be specified.
