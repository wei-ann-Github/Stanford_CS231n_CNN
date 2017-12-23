Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #3 of Spring 2017.
This assignment consists of 5 exercises.

**Exercise on Image Captioning with Vanilla RNNS**:
* Introduced to the [Microsoft COCO dataset](http://cocodataset.org/#home) of images with labelled captions. This dataset will be used for the assessment of the image captioning algorithm using vanilla RNN
* Learn how the gradients are propagated through the RNN.
* Implement an image captioning RNN using numpy.
* Shows how the image to be captioned in fed into the RNN.
* Demonstrates how training and inference are implemented differently in the RNN.
* How to deal with varying caption length in the captioning task.

**Exercise on Image Captioning with LSTMs**:
* Learnt about the LSTM architecture and equations and how to implement it in numpy.
* Derive the gradients in LSTM.
* Introduced to BLEU for quantitatively evaluating the quality of the captioning.
* Train and tune an LSTM model to perform the image captioning task.

**Exercise on Network Visualization: Saliency Maps, Class Visualization, and Fooling Images**:
* Introduced to Squeezenet and its architecture.
* Showed that gradients are not only generated at each of the weights, gradients are also generated at the pixel level, and this gradient can be used to perform "updates" to the image's pixel to different effect.
* Build a Saliency Map for each sample image to visualize which image pixel is influential in affecting the model's prediction.
* How to modify images that can fool a pretrained CNN to make wrong prediction of the images' class, when to the human eye, the images looked the same.
* Generate "images" of a class in the pretrained CNN. The "images" generated are generally not very clear, to the human eye, what the network is trying to depict.

**Exercise on Style Transfer**:
* Understand how style transfer works (obviously) and how it is different from the usual classification CNN. It requires a content image, a style image, and a generated image.
* Introduced and implementing the content loss function in either PyTorch or Tensorflow.
* Introduced and implement the style loss function and Gram Matrix required to calculate the style loss function.
* Introduced and implement the total-variation regularizer as the regularizer to smooth the generated image.
* Introduced and implement the style transfer overall loss function.
* Perform the style transfer.

**Exercise on Generative Adversarial Networks**
* Build a Generative Adversarial Network (GAN) based on the MNIST dataset.
* Provided a very clear explanation of the concept of GAN.
* Provided a good number of reference to past and continuous development in GAN.
* Introduced to the Leaky Relu.
* Building a discriminator network and a generative network.
* Build different discriminator and generator loss function such as that from Least Square GAN and DCGAN. 
* Can also try your hand at building the loss functions from the [Improved Wasserstein GAN](https://arxiv.org/abs/1704.00028)

