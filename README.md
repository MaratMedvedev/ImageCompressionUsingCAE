# About
Hello, I would like to introduce my project "Image compression using neural networks". At the beginning, I should say that it is more research project. So, I try to figure out ways how I can improve compression images using neural networks. I used the mnist dataset to check the results.

## Solution
I train convolutional autoencoder (model located in the `CAE.py` file) on mnist dataset. To compress, I pass all the images through the encoder to get the feature vectors and save them to a file. I also save decoder weights to a file. After that, I compress the file with the codes and with the decoder into a zip file. Finally, this zip can be passed to another user. To decompress data, it’s required to apply a decoder to each code and recover the dataset.

## Results
To find out memory consumption and quality of files you can check the following files:  
Original data: `mnist.zip`  
The compressed data: `mnist_compressed.zip`  
The compressed and decompressed data: `mnist_compressed_decompressed.zip`

## Conclusion
Convolutional autoencoder compression is helpful for some simple specialized data (like the MNIST dataset). However, for more complex data, this method is less effective, because significant lossiness happens when compression is applied.

# How run the program?
## Compressor
To compress file you can just run the `CAE_compressor.py` from command line with two parameters:  
**First parameter** is path to file that you want compress  
**Second parameter** is the maximum time that the program will run (the restriction helps to save time, but worsens the quality of the images)  
**Time format:** {number}h {number}m {number}s  

**Example:**  
  *Command line:* `python CAE_compressor.py mnist.zip 2h 15m 30s`

## Decompressor
To decompress file you can just run the `CAE_decompressor.py` from command line with two parameters:  
**First parameter** is path to file that you want decompress  
**Second parameter** is folder that will contain results of decompressing (optional parameter)

**Example:**  
  *Command line:* `python CAE_decompressor.py mnist.zip mnist_dataset_images`
