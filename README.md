# About
Hello, I would like to introduce my project ‘Image compression using neural networks’’. At the beginning, I should say that it is more research project. So, I try to figure out ways how I can improve compression images using neural networks. I used the mnist dataset to check the results.

### Solution
I train convolutional autoencoder on mnist dataset. To compress, I pass all the images through the encoder to get the feature vectors and save them to a file. I also save decoder weights to a file. After that, I compress the file with the codes and with the decoder into a zip file. Finally, this zip  can be passed to another user. To decompress data, it’s required to apply a decoder to each code and recover the dataset.

### Results
To find out memory consumption and quality of files you can check the following files:  
Original data: mnist.zip  
The compressed data: mnist_compressed.zip  
The compressed and decompressed data: mnist_compressed_decompressed.zip

### Conclusion
Convolutional autoencoder compression is helpful for some simple specialized data (like the MNIST dataset). However, for more complex data, this method is less effective, because significant lossiness happens when compression is applied, .

# How run the program?
