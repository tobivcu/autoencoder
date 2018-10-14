# Autoencoder for Compression and Decompression
Authors: Tong Liu and Shakeel Alibhai

## Overview
This is an autoencoder that uses machine learning to compress and decompress files. It first takes a training file that is used to train the file (ie. generating weight matrices and bias vectors). It will then save the weight matrices and bias vectors so that they can be used for compression and/or decompression.

## Usage
### Setting the Parameters
There are several parameters that can be modified in the program. These include, but are not necessarily limited to, the following:

* Number of Epochs
* Learning Rate/Learning Rate Decay
* Batch Size
* n_input

### Specifying a Compression Ratio
Currently, the only way to specify a theoretical compression ratio is to modify the code. The compression ratio is equal to the product of n_input, n_hidden1, n_hidden2, and n_hidden3.
Note, however, that this is only a theoretical compression ratio. The actual compression ratio may be lower than the theoretical compression ratio. This may occur for several reasons, such as the range of the input data and how well the autoencoder is optimized to predict the data.

### Running the Autoencoder
Arguments: -r for training file; -c for compression file; -d for decompression file; -e for error-bound

* Example with "train.txt" as a training file and "data.txt" as a compression file: python3 autoencoder.py -r train.txt -c data.txt
* Example with "p.txt" as a file to decompress: python3 autoencoder.py -d p.txt (Note: In this case, "p.txt" must be the output file from the compression step of the autoencoder.)

Note: Compression and decompression cannot be done together!

Note: Decompression may currently not fully work yet.