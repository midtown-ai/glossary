---
title: V
permalink: /v/

#=== THEMES
# = minima
layout: page
# = minimal mistake
# layout: archive-taxonomy
# layout: archive
# layout: categories
# layout: category
# layout: collection
# layout: compress
# layout: default
# layout: home
# layout: posts
# layout: search
## layout: single
# layout: splash
# layout: tag
# layout: tags

---

* toc
{:toc}

{% include links/all.md %}


# Vanilla GAN

 The Vanilla GAN is the simplest type of GAN made up of the generator and discriminator , where the classification and generation of images is done by the generator and discriminator internally with the use of multi layer perceptrons. The generator captures the data distribution meanwhile , the discriminator tries to find the probability of the input belonging to a certain class, finally the feedback is sent to both the generator and discriminator after calculating the loss function , and hence the effort to minimize the loss comes into picture.

 ![]( {{site.assets}}/v/vanilla_gan.png ){: width="100%"}

 See also [V], [Generative Adversarial Network]


# Vanishing Gradient Problem

  = a problem that arises because of the loss of information in backpropagation with forward activation function (sigmoid, etc). Vanishing Gradient Problem is a difficulty found in training certain Artificial Neural Networks with gradient based methods (e.g Back Propagation). In particular, this problem makes it really hard to learn and tune the parameters of the earlier layers in the network. This problem becomes worse as the number of layers in the architecture increases. :warning: This is not a fundamental problem with neural networks - it's a problem with gradient based learning methods caused by certain activation functions. Let's try to intuitively understand the problem and the cause behind it. 
   * Problem ==> Gradient based methods learn a parameter's value by understanding how a small change in the parameter's value will affect the network's output. If a change in the parameter's value causes very small change in the network's output - the network just can't learn the parameter effectively, which is a problem. This is exactly what's happening in the vanishing gradient problem -- the gradients of the network's output with respect to the parameters in the early layers become extremely small. That's a fancy way of saying that even a large change in the value of parameters for the early layers doesn't have a big effect on the output. Let's try to understand when and why does this problem happen. 
   * Cause ==> Vanishing gradient problem depends on the choice of the activation function. Many common activation functions (e.g sigmoid or tanh) 'squash' their input into a very small output range in a very non-linear fashion. For example, sigmoid maps the real number line onto a "small" range of [0, 1], especially with the function being very flat on most of the number-line. As a result, there are large regions of the input space which are mapped to an extremely small range. In these regions of the input space, even a large change in the input will produce a small change in the output - hence the gradient is small. :warning: This becomes much worse when we stack multiple layers of such non-linearities on top of each other. For instance, first layer will map a large input region to a smaller output region, which will be mapped to an even smaller region by the second layer, which will be mapped to an even smaller region by the third layer and so on. As a result, even a large change in the parameters of the first layer doesn't change the output much.

 To minimize this problem, you can try to
  * use the ReLU activation function over the sigmoid ones. ReLU which NOT cause a small derivative if >= 0.
  * reduce the number of layers in the network (minimize total loss by reducing the number of times the signal goes through an activation function), 
  * use batch normalization (don't reach the outer edges of the sigmoid function) = work in a regime (input value range) where the derivative is not zero 
  * change model architecture
  * and/or use residual networks as they provide residual connections straight to earlier layers. The residual connection directly adds the value at the beginning of the block, x, to the end of the block (F(x)+x). This residual connection doesn???t go through activation functions that ???squashes??? the derivatives, resulting in a higher overall derivative of the block.

 More at 
  * [https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
  * [https://www.quora.com/What-is-the-vanishing-gradient-problem](https://www.quora.com/What-is-the-vanishing-gradient-problem)
  * [https://en.wikipedia.org/wiki/Vanishing_gradient_problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

 See also [V], [Activation Function], [Batch Normalization], [Exploding Gradient Problem], [Rectified Linear Unit], [Residual Network Model]


# Variable

 See also [V], ...


# Variable Model

 See also [V], [Bayesian Network], [Constraint Satisfaction Problem], [Model Type]


# Variable Type

  * Continuous variable
  * Categorical variable
  * Discrete variable

 See also [V], [Categorical Variable], [Continuous Variable], [Discrete Variable]

# Variance

 How dispersed your predicted values are. Low variance high bias = underfitting. High variance + low bias = overfitting.

 See also [V], [Bias], [Overfitting], [Underfitting], 


# Variational Autoencoder

# VAE

 VAEs are autoencoders (encoder + latent space + decoder) that encode inputs as distributions instead of points and whose latent space ???organisation??? is regularised by constraining distributions returned by the encoder to be close to a standard Gaussian. :warning: In the Autoencoder bottleneck, you have 2 vectors: (1) the mean vector, (2) the variance vector of the distributions. The input of the decoder is a sample of the distributions.

 ![]( {{site.assets}}/v/variational_autoencoder.png ){: width="100%"}

  * first, the input is encoded as distribution over the latent space
  * second, a point from the latent space is sampled from that distribution
  * third, the sampled point is decoded and the reconstruction error can be computed
  * finally, the reconstruction error is backpropagated through the network 


 ![]( {{site.assets}}/v/autoencoder_type.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=9zKuYvjFFS8" %}

 Why are VAE better than simple autoencoder? ==> making the generative process possible 
 
 ```
# VAE LOSS FUNCTION = RECONSTRUCTION LOSS - KL DIVERGENCE
 ```

 ![]( {{site.assets}}/v/variational_autoencoder_loss_function_formula.png ){: width="100%"}

 Beware:
  * backpropagation cannot be done with VAE, because of the sampling between the encoder and decoder. The solution is to use the "reparameterization trick"

 More at:
  * [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73 ](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
  * [https://jaan.io/what-is-variational-autoencoder-vae-tutorial/](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

 See also [V], [Autoencoder], [Autoencoder Type], [Disentangled Variational Autoencoder], [Generative Model], [Kullback-Leibler Divergence], [Latent Space], [Variational Autoencoder Reparameterization Trick], [Vector Quantized Variational Autoencoder]


# Variational Autoencoder Reparameterization Trick

 Because in a variational autoencoder, you sample the output of the encoder to feed the decoder, you cannot use backpropagation. The solution to this is to use this reparametrization trick.

 ![]( {{site.assets}}/v/variational_autoencoder_reparametrization_trick.png ){: width="100%"}

 More at:
  * [https://youtu.be/rZufA635dq4?t=1401](https://youtu.be/rZufA635dq4?t=1401)
  * [https://towardsdatascience.com/reparameterization-trick-126062cfd3c3](https://towardsdatascience.com/reparameterization-trick-126062cfd3c3)

 See also [V], [Backpropagation], [Deterministic Node], [Stochastic Node], [Variational Autoencoder]


# Vector Quantized Variational Autoencoder

# VQ-VAE

 VQ-VAE extends the standard autoencoder by adding a discrete codebook component to the network. The codebook is basically a list of vectors associated with a corresponding index.

 ![]( {{site.assets}}/v/vector_quantized_variational_autoencoder.png ){: width="100%"}

 It is used to quantize the bottleneck of the autoencoder; the output of the encoder network is compared to all the vectors in the codebook, and the codebook vector closest in euclidean distance is fed to the decoder. Mathematically this is written as 
 
 ```
z_q(x)=\text{argmin}_i ||z_e(x)-e_i||_2 
# where z_e(x) is the encoder vector for some raw input x
# e_i is the ith codebook vector
# and z_q(x) is the resulting quantized vector that is passed as input to the decoder.
 ```

 This argmin operation is a bit concerning, since it is non-differentiable with respect to the encoder. But in practice everything seems to work fine if you just pass the decoder gradient directly through this operation to the encoder (i.e. set its gradient to 1 wrt the encoder and the quantized codebook vector; and to 0 wrt all other codebook vectors). The decoder is then tasked with reconstructing the input from this quantized vector as in the standard autoencoder formulation.

 More at :
  * [https://ml.berkeley.edu/blog/posts/vq-vae/](https://ml.berkeley.edu/blog/posts/vq-vae/)

 See also [V], [Codebook], [Variational Autoencoder]


# Vector

 A 1 column matrix (akak a list!) that represent all the inputs to a neural network or a summary of all the values of the features. Not a tensor (matrix).

 See also [V], [Dot Product], [Feature], [Tensor], [Sparse Vector], [Vector Database]


# Vector Database

 Being able to search across images, video, text, audio, and other forms of unstructured data via their content rather than human-generated labels or tags is exactly what vector databases were meant to solve. When combined with powerful machine learning models, vector databases such as Milvus have the ability to revolutionize e-commerce solutions, recommendation systems, computer security, pharmaceuticals, and many other industries. A vector database is a fully managed, no-frills solution for storing, indexing, and searching across a massive dataset of unstructured data that leverages the power of embeddings from machine learning models. A vector database should have the following features:
  * scalability and tunability,
  * multi-tenancy and data isolation,
  * a complete suite of APIs, and
  * an intuitive user interface/administrative console.

 See also [V], [Milvus Database], [Representation Space], [Vector], [Vector Search Library]


# Vector Search Library

 projects such as FAISS, ScaNN, and HNSW are lightweight ANN libraries rather than managed solutions. The intention of these libraries is to aid in the construction of vector indices ??? data structures designed to significantly speed up nearest neighbor search for multi-dimensional vectors. If your dataset is small and limited, these libraries can prove to be sufficient for unstructured data processing, even for systems running in production. However, as dataset sizes increase and more users are onboarded, the problem of scale becomes increasingly difficult to solve. Vector databases also operate in a totally different layer of abstraction from vector search libraries - vector databases are full-fledged services, while ANN libraries are meant to be integrated into the application that you???re developing. In this sense, ANN libraries are one of the many components that vector databases are built on top of, similar to how Elasticsearch is built on top of Apache Lucene.

 See also [V], [Vector], [Vector Database]


# Vector Search Plugin

 An increasing number of traditional databases and search systems, such as Clickhouse and Elasticsearch, include built-in vector search plugins. Elasticsearch 8.0, for example, includes vector insertion and ANN search functionality that can be called via restful API endpoints. The problem with vector search plugins should be clear as night and day - these solutions do not take a full-stack approach to embedding management and vector search. Instead, these plugins are meant to be enhancements on top of existing architectures, thereby making them limited and unoptimized. Developing an unstructured data application atop a traditional database would be like trying to fit lithium batteries and electric motors inside a frame of a gas-powered car - not a great idea! To illustrate why this is, let???s go back to the list of features that a vector database should implement (from the first section). Vector search plugins are missing two of these features - tunability and user-friendly APIs/SDKs.
 See also [V], [Vector], [Vector Database]


# Video Pre-Training Model

# VPT Model

 {% youtube "https://www.youtube.com/watch?v=oz5yZc9ULAc" %}

 {% youtube "https://www.youtube.com/watch?v=ODat7kfZ-5k" %}

 More at:
  * paper - [https://arxiv.org/abs/2206.11795](https://arxiv.org/abs/2206.11795)
  * code - [https://github.com/openai/Video-Pre-Training](https://github.com/openai/Video-Pre-Training)
  * blog post - [https://openai.com/blog/vpt/](https://openai.com/blog/vpt/)

 See also [V], [Inverse Dynamics Model], [Reinforcement Learning]


# VIMA Model

 {% youtube "https://www.youtube.com/watch?v=oLBg8aLoQ00" %}

 See also [V], [Nvidia Company]


# Visual Geometry Group Model

# VGG-16 Model

# VGG-19 Model

# VGG Model

 A model developed by VGG in the Department of Engineering Science, University of Oxford. 
  * VGG-19 = The number 19 stands for the number of layers with trainable weights. 16 Convolutional layers and 3 Fully Connected layers. The VGG-19 was trained on the ImageNet challenge (ILSVRC) 1000-class classification task. The network takes a (224, 224, 3) RBG image as the input.

 More at
  * [https://medium.com/mlearning-ai/image-detection-using-convolutional-neural-networks-89c9e21fffa3](https://medium.com/mlearning-ai/image-detection-using-convolutional-neural-networks-89c9e21fffa3)
  * [https://www.image-net.org/challenges/LSVRC/](https://www.image-net.org/challenges/LSVRC/)

 See also [V], [Convoluted Neural Network]


# Virtual Assistant

 All the Alexas, Siris, Google Assistants, and customer support chatbots of the world fall into this category. They use NLP to understand, analyze, and prioritize user questions and requests, and respond to them quickly and correctly.

 See also [V], [Natural Language Processing]


# Vision-Language Pre-Training

# VLP

 {% pdf "{{site.assets}}/v/vision_language_pretraining_paper.pdf" %}

 See also [V], [Masked Vision Modeling]


# Vision Transformer

# ViT

 Used to caption images! Trained on imagNet. Instead of a tokenizer, uses a feature_extractor (image kernels? No, the whole image).

 The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable ???classification token??? to the sequence is used.

 {% pdf "{{site.assets}}/v/vision_transformer_paper.pdf" %}

 ![]( {{site.assets}}/v/vision_transformer_architecture.png ){: width="100%"}

 See also [V], [Feature Extractor], [Tokenizer]


# Visual Grounding

 Visual grounding is the task of localizing concepts referred to by the language onto an image. 


# Voice Encoder

# Vocoder

 used to transform the generated mel-spectrogram into a waveform.

 {% youtube "https://www.youtube.com/watch?v=2Iq5658IlFc" %}

 More at:
  * [https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

 See also [V], [Encoder]
