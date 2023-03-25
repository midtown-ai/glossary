---
title: D
permalink: /d/

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


# DALL-E Model

 A play on words between WALL-E and Dali!

 More at :
  * open-ai announcement - [https://openai.com/dall-e-2/](https://openai.com/dall-e-2/)
  * site + paper - [https://openai.com/blog/dall-e/](https://openai.com/blog/dall-e/)
  * how does DALL-E work? - [https://www.assemblyai.com/blog/how-dall-e-2-actually-works/](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)
  * DALL-E 2 uses CLIP - [https://arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)

 See also [D], [CLIP Model], [GLIDE Model]


# DARPA

 * [DARPA Grand Challenge]
 * [DARPA Siri]
 * [DARPA Robotics Challenge]
 * [DARPA Subterranean Challenge]

 More at:
  * ...

 See also [D], ...


# DARPA Robotics Challenge

 2015 Challenge

 {% youtube "https://www.youtube.com/watch?v=8P9geWwi9e0" %}

 See also [D], [AI Challenge]


# DARPA Grand Challenge

 2004 and 2005 Driverless car competition. No one completed the challenge in 2004. So second challenge in 2005!
 123 miles in the desert.
 Time trial.
 $1 million grand prize in 2004.
 $2 million grand prize in 2005.
  * (1) best time: Stanford with Stanley
  * (2)
  * (3)
  * (4) 

 {% youtube "https://www.youtube.com/watch?v=7a6GrKqOxeU" %}

 {% youtube "https://www.youtube.com/watch?v=TDqzyd7fDRc" %}

 More at:
  * 2004 highlights - [https://www.youtube.com/watch?v=P__fbWm6wlg](https://www.youtube.com/watch?v=P__fbWm6wlg) 

 See also [D], [AI Challenge], [Autonomous Vehicle], [DARPA Urban Challenge], [LIDAR]


# DARPA Subterranean Challenge
 
 In 2021. Map, navigate, and search for object/people.
  * system competition
   * $2 million - CEREBUS team
   * $1 million - CSIRO 61 Team
   * $500K - Marble Team
  * virtual competition
   * $750K
   * $500K
   * $250K

  {% youtube "https://www.youtube.com/watch?v=HuJGIAjuxLE" %}

  {% youtube "https://www.youtube.com/watch?v=6OB7r4gUh74" %}

  More at:
   * [https://subtchallenge.com/](https://subtchallenge.com/)

  See also [D], [AI Challenge]


# DARPA Urban Challenge

 2007 Driverless car competition. Moving traffic.
  * first place: BOSS
  * second place: Junior/Stanford/Silicon Valley
  * third place: ..

 {% youtube "https://www.youtube.com/watch?v=aHYRtOvSx-M" %}

 More at:
  * 

 See also [D], [AI Challenge], [Autonomous Vehicle], [DARPA Grand Challenge], [LIDAR]


# Data

 The rocket fuel for AI, ML, etc.

 See also [D], [Data Augmentation], [Data Normalisation]


# Data Analyst

 Focused on the tools.

 See also [D], [Data Scientist]


# Data Augmentation

 To use when you don't have enough data.
  * For images, you can increase the number of samples by
   * flipping the image
   * zooming on the image
   * moving the image
   * cropping the image
  * With voice, you can
   * change the speech speed
   * change the volume
   * change the pitch.

 Beware:
   * :warning: The transformation operation should be invariant and not change the output.

 See also [D], [Data], [Insufficient Data Algorithm], [Self-Supervised Learning], [Snorkel Program], [Zero-Shot Learning]


# Data Handling

 See also [D], [Hyperparameter]


# Data Leakage

 Data leakage is an umbrella term covering all cases where data that shouldn’t be available to a model in fact is. The most common example is when test data is included in the training set. But the leakage can be more pernicious: when the model uses features that are a proxy of the outcome variable or when test data come from a distribution which is different from the one about which the scientific claim is made.

 More at:
   * [https://docs.google.com/presentation/d/1WrkeJ9-CjuotTXoa4ZZlB3UPBXpxe4B3FMs9R9tn34I/edit#slide=id.g164b1bac824_0_2980](https://docs.google.com/presentation/d/1WrkeJ9-CjuotTXoa4ZZlB3UPBXpxe4B3FMs9R9tn34I/edit#slide=id.g164b1bac824_0_2980)


# Data Normalisation

 Cleaning the data in preparation of feeding it to a model.

 See also [D], [Data]


# Data Point

 ~ an observation. Together the features and the label make a single data point. Imputation is a way to deal with missing data in data points.
 
 ![]( {{site.assets}}/d/data_point.png ){: width="100%"}

 See also [D], [Dataset], [Feature], [Imputation], [Label]


# Data Science

 ![]( {{site.assets}}/d/data_science.png ){: width="100%"}

 See also [D], ...


# Data Scientist

 Choose the toolset, not the cloud provider and database type!
  * data acquisition
  * data manipulation
  * data movement
  * ...

 See also [D], [Data Analyst], [DevOps]


# Data2Vec

 A General Framework for Self-supervised Learning in Speech, Vision and Language. While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

 {% pdf "{{site.assets}}/d/data2vec_paper.pdf" %}

 More at:
   * paper - [https://arxiv.org/abs/2202.03555](https://arxiv.org/abs/2202.03555)

 See also [D], [Self-Supervised Learning]


# Dataset

 A group of several observations. Good data must contain a signal of what you are trying to measure. Beware that data-set may be incomplete. For example, if you are trying to forecast inventory, you can look at sale's number, but the data needs to includes the times when the sale did not happen because we run out of inventory! Dataset needs to be at least 10 times the number of features. The dataset is split in 3 subsets called the training subset, the development subset, and the test subset. If you have a lot of data 70% goes to the training, 15% to the dev, and 15% to the test. If not much data, 80% goes to training, 10% and 10%. A dataset is based on either one of those:
  * images (MNIST for digits, CIFAR-10 for 10 categories, ImageNet)
  * audio sequences
  * text, aka corpus such as as C4
  * sentences
  * words (!WordNet, GSM8K for math reasoning)

 More at
  * [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
  * [https://paperswithcode.com/datasets](https://paperswithcode.com/datasets)
  * [https://scikit-learn.org/stable/datasets.html](https://scikit-learn.org/stable/datasets.html)

 See also [D], [Corpus], [CIFAR-10 Dataset], [Data Point], [Development Subset], [GINI Impurity Index], [ImageNet Dataset], [MNIST Dataset], [Testing Subset], [Training Subset], [WordNet Dataset]


# Dawid-Skene Algorithm

 When you crowdsource a labeling task, how can you be certain that the label is correct? Have several people label the same image/entry and apply this algorithm! An alternative is to use majority vote algorithm.

 More at:
  * sample coded example - [https://github.com/Ekeany/Dawid-Skene](https://github.com/Ekeany/Dawid-Skene)
  * fast dawid-skene paper - [https://deepai.org/publication/fast-dawid-skene](https://deepai.org/publication/fast-dawid-skene)

 See also [D], [Labeling Service], [Majority Vote Algorithm]


# DBSCAN

 To use for clustering when k-mean fails. With k-Mean, we look for round clusters. With DBSCAN, the radius (epsilon) is from every point in the cluster ==> the cluster shape does not need to be round! If epion is too large --> gigantic cluster. If too small, --> ...

 See also [D], [K-Mean Failure]


# Decision Boundary

 Can be a circle, a square, a plane and is used in a decision tree. In fact, it can be any shape! A neural netowkr will use the weights and activation function to lower the loss function to find the "perfect" boundary.

 ![]( {{site.assets}}/d/decision_boundary.png ){: width="100%"}

 See also [D], [Activation Function], [Classification], [Decision Tree], [Hyperplane], [Logistic Regression]


# Decision Stump

 A decision tree with only one split.

 See also [D], [AdaBoost], [Decision Tree]


# Decision Tree

 Decision trees are [White Box Models][White Box Model] that Can be used for regression and classification.
  * classification:  Go from the root node to the leaf of the tree where is the classification.
  * regression: use the mean square error (MSE)

 More at:
  * regression tree - [https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047](https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047)

 See also [D], [Classification], [Decision Stump], [Regression]


# Decoder
  * Use masked attention ( = only access to a single context, i.e. right or left context)
 Good for
  * natural language generation, generative AI

 {% youtube "https://www.youtube.com/watch?v=d_ixlCubqQw" %}

 See also [D], [Autoregressive], [Decoder Stack], [Encoder], [Encoder-Decoder Model], [Hidden State], [Image Decoder], [Masked Attention], [Natural Language Generation]


# Decoder Representation
 
 This is ...

 See also [D], ...


# Decoder Representation Space

  * Pixel space

 See also [D], [Encoder Representation Space], [Representation Space]


# Decoder Stack

 See also [D], [Decoder], [Encoder-Decoder Model], [Encoder Stack], [Hidden State]


# Deconvolution Neural Network

 The decoder part of a convolutional autoencoder.

 See also [D], [Convolution Autoencoder], [Convolution Neural Network]


# Deductive Reasoning

 Sherlock Holmes!

 See also [D], [Inductive Reasoning]


# Deep Belief

 A type of neural network.

 See also [D], [Neural Network]


# Deep Belief Network

# DBN

 See also [D], [Boltzmann Machine]


# Deep Blue Challenge

 Garry Kasparov vs. Deep Blue in 1996 and 1997. Deep blue is an [heuristic-basedi][Heuristic] game-playing program.

 {% youtube "https://www.youtube.com/watch?v=KF6sLCeBj0s" %}

 {% youtube "https://www.youtube.com/watch?v=mX_fQyxKiPQ" %}

 More at:
  * [https://www.sciencedirect.com/science/article/pii/S0004370201001291](https://www.sciencedirect.com/science/article/pii/S0004370201001291)

 See also [D], [AI Challenge]


# Deep Convolutional GAN

# DCGAN

 A type of GAN for ... This is the first GAN where the generator used deep convolutional network , hence generating high resolution and quality images to be differentiated. Rectified Linear Unit (ReLU) activation is used in Generator all layers except last one where Tanh activation is used, meanwhile in Discriminator all layers use the Leaky-ReLu activation function. Adam optimizer is used with a learning rate of 0.0002.

 ![]( {{site.assets}}/d/deep_convolutional_gan.jpeg ){: width="100%"}

 The above figure shows the architecture of generator of the GAN. The input generated is of 64 X 64 resolution.

 See also [D], [Generative Adversarial Network], [Rectified Linear Unit]


# Deepfake

 Deepfakes (a portmanteau of "deep learning" and "fake") are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. While the act of creating fake content is not new, deepfakes leverage powerful techniques from machine learning and artificial intelligence to manipulate or generate visual and audio content that can more easily deceive. The main machine learning methods used to create deepfakes are based on deep learning and involve training generative neural network architectures, such as autoencoders, or generative adversarial networks (GANs).

 Deepfakes have garnered widespread attention for their potential use in creating child sexual abuse material, celebrity pornographic videos, revenge porn, fake news, hoaxes, bullying, and financial fraud. This has elicited responses from both industry and government to detect and limit their use.

 From traditional entertainment to gaming, deepfake technology has evolved to be increasingly convincing and available to the public, allowing the disruption of the entertainment and media industries.

 {% youtube "https://www.youtube.com/watch?v=Yb1GCjmw8_8" %}

 More at:
  * [https://en.wikipedia.org/wiki/Deepfake](https://en.wikipedia.org/wiki/Deepfake)

 See also [D], [AI Avatar]

# Deep Learning

 A branch of AI, a sub branch of Machine learning with neural networks! Use layers of non-linear processing units for feature extraction and transformation. Each layer use the output from the previous layer. May be supervised or unsupervised learning. Applications include pattern analysis (unsupervised) or classification (supervised or unsupervised).

 See also [D], [Deep Learning Framework], [Machine Learning], [Percepton]


# Deep Learning Framework

 From deep learning revolution that started ~ 2007.

 See also [D], [Caffe], [MXNET], [PyTorch ML Framework], [TensorFlow ML Framework]


# Deep Multi-Task Learning

 See also [D], ...


# Deep Neural Network

# DNN

 A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers. The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output.


# Deep Reinforcement Learning

 See [Reinforcement Learning]


# DeepAR Forecasting

 Based on neural network. `Time series forecasting` (ex number of units sold). Model needs to be trained, i.e. supervised. Integrated with Sagemaker. Lots of hyperparameters. Tuning is very important.


# DeepMind Company

 Models:
  * [AlphaCode][AlphaCode Model]: LLM for code generation
  * [AlphaFold][AlphaFold Model]: Protein folding
  * [AlphaGo][AlphaGo Model]: Agent to play Go
  * [AlphaStar][AlphaStar Model]: Agents to play StarCraft 2
  * [AlphaTensor][AlphaTensor Model]: Matrix multiplication algorithm optimization
  * [AlphaZero][AlphaZero Model]
  * [Chinchilla][Chinchilla Model]: Optimized version of the [Gopher Model]
  * [Gato][Gato Model]: Multi-task generalist agent
  * [Gopher][Gopher Model]: A LLM with same (or better) performance than GPT3
  * [Sparrow][Sparrow Model]: A ChatGPT alternative

 {% youtube "https://www.youtube.com/watch?v=kFlLzFuslfQ" %}

 More at :
  * [https://www.deepmind.com/research](https://www.deepmind.com/research)
  * publications - [https://www.deepmind.com/research/publications](https://www.deepmind.com/research/publications)

 See also [D], ...


# DeepSpeed Project

 {% youtube "https://www.youtube.com/watch?v=pDGI668pNg0" %}

 More at :
  * [https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

 See also [D], [Microsoft Company]


# Delayed Reward

 `You must reward for the correct outcome!` Do not only reward for the completion of an assignment, but for passing the final exam. Ex: In chess, what matters is winning the game, not really how many piece you have kept at the end!

 See also [D], [Addiction], [Reinforcement Learning], [Reward Shaping]


# Dendrite

 See also [D], [Biological Neuron]


# Denoising Autoencoder

 ```
input + noise --> ENCODER --(latent space)--> DECODER ---> output

with loss function computed from (output - input)
==> the autoencoder tries to remove the noise from the original image!
 ```

 See also [D], [Autoencoder], [Loss Function]


# Denoising Diffusion Probabilistic Model

 See [Diffusion Model]


# Dense Layer

 In any neural network, a dense layer is a layer that is deeply connected with its preceding layer which means the neurons of the layer are connected to every neuron of its preceding layer. This layer is the most commonly used layer in artificial neural network networks. As discussed before, results from every neuron of the preceding layers go to every single neuron of the dense layer. So we can say that if the preceding layer outputs a (M x N) matrix by combining results from every neuron, this output goes through the dense layer where the count of neurons in a dense layer should be N.

 See also [D], [Dense Model], [Discriminator], [U-Net Architecture]


# Dense Model

 Most of today’s models are “dense,” which means the whole neural network activates to accomplish a task, regardless of whether it’s very simple or really complicated.

 See also [D], [Dense Layer]


# Derivative Chain Rule

 A fundamental rule that is used to do backpropagation starting from the cost function.

 ![]( {{site.assets}}/d/derivative_chain_rule.png ){: width="100%"}

 ```
aL = activation/output at layer L
bL = bias at layer/percepton L
wL = weight at layer L

zL = wL * aL-1 + bL
aL = activation_function(zL)          <== zero activated as bias is already in zL

Cost = ( aL - yDESIRED)^2             <== yDESIRED determines the weights

==> dC/dwL  == change/nudge in cost/loss function due to a change/nudge in wL (weight)
but using chain rule, this is easy to compute as

Let's start with 1 sample
dCk/daL = 2 ( aL - yDESIRED)                              <== k is index or sample number
daL/dzL = derivative_of_activation_function ( zL )
dzL/dwL = aL-1

Now with all the samples
dC/dwL =  sum(0,nb_samples-1, dCk/dwL) / nb_samples

We need to do this for
* dC/daL-1                     <== backpropagate
* dC/dbL                       <== compute the bias L as well with backpropagation !!!!
 ```

 See also [D], [Backward Propagation], [Loss Function]


# Deterministic Node

 A node whose output is the same given a set of inputs. Works unlike a stochastic node! 

 See also [D], [Stochastic Node], [Variational Autoencoder Reparametrization Trick]


# Development Subset

 Use to test the model built with the training set before it is run on the test subset.

 See also [D], [Dataset], [Test Subset], [Training Subset]


# DevOPS

 See also [D], [Data Analyst], [Data Scientist]


# Differential Privacy

 Differential privacy is the technology that enables researchers and database analysts to avail a facility in obtaining the useful information from the databases, containing people's personal information, without divulging the personal identification about individuals. This can be achieved by introducing a minimum distraction in the information, given by the database.

 Examples:
  * Apple employs differential privacy to accumulate anonymous usage insights from devices like iPhones, iPads and Mac.
  * Amazon uses differential privacy to access user’s personalized shopping preferences while covering sensitive information regarding their past purchases.
  * Facebook uses it to gather behavioral data for target advertising campaigns without defying  any nation’s privacy policies.

 For example, consider an algorithm that analyzes a dataset and compute its statistics such as mean, median, mode, etc. Now, this algorithm can be considered as differentially private only if via examining at the output if a person cannot state whether any individual’s data was included in the actual dataset or not.

 In simplest form, the differentially private algorithm assures that there is hardly a behaviour change when an individual enlists or moves the datasets. Or simply, the algorithm might produce an output, on the database that contains some individual’s information, is almost the same output that a database generates without having individuals’ information. This assurance holds true for any individual or any dataset. 

 Thus, regardless of how particular an individual’s information is, of the details of any other person in the database, the guarantee of differential privacy holds true and provides a formal assurance that individual-level information about participants in the database would be preserved, or not leaked.

 ![]( {{site.assets}}/d/differential_privacy.png ){: width="100%"}

 Methods:
  * output perturbation (works for all cases since treat the model as a black box?)
  * gradient perturbation (works for neural network only)

 More at:
  * [https://www.analyticssteps.com/blogs/what-differential-privacy-and-how-does-it-work](https://www.analyticssteps.com/blogs/what-differential-privacy-and-how-does-it-work)

 See also [D], [Gradient Perturbation], [Membership Inference Attack], [Output Perturbation]


# Diffusion Model

# DM

 There are a few downsides to Diffusion models: they work sequentially on the whole image, meaning that both the training and inference times are expansive. This is why you need hundreds of GPUs to train such a model and why you wait a few minutes to get your results. they are iterative models that take random noise as inputs, which can be conditioned with a text or an image, so it is not completely random noise. It iteratively learns to remove this noise by learning what parameters the model should apply to this noise to end up with a final image. So the basic diffusion models will take random noise with the size of the image and learn to apply even further noise until we get back to a real image.

 ![]( {{site.assets}}/d/diffusion_model.png ){: width="100%"}

 This is possible because the model will have access to the real images during training and will be able to learn the right parameters by applying such noise to the image iteratively until it reaches complete noise and is unrecognizable. Then, when we are satisfied with the noise we get from all images, meaning that they are similar and generate noise from a similar distribution, we are ready to use our model in reverse and feed it similar noise in the reverse order to expect an image similar to the ones used during training.

 {% youtube "https://www.youtube.com/watch?v=W-O7AZNzbzQ" %}

 More at:
  * what are diffusion models - [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  * diffusion thermo model - [https://arxiv.org/pdf/1503.03585.pdf](https://arxiv.org/pdf/1503.03585.pdf)
  * Diffusion Models Beat GANs on Image Synthesis - [https://arxiv.org/pdf/2105.05233.pdf](https://arxiv.org/pdf/2105.05233.pdf)
  * Denoising Diffusion Probabilistic Models - [https://arxiv.org/pdf/2006.11239.pdf](https://arxiv.org/pdf/2006.11239.pdf)
  * Improved Denoising Diffusion Probabilistic Models - [https://arxiv.org/pdf/2102.09672.pdf](https://arxiv.org/pdf/2102.09672.pdf)

 See also [D], [Generative Model], [Latent Diffusion Model]


# Diffusion Process

 Coming from the real pgysical diffusion process, but for adding noise to an image.

 See also [D], [Diffusion Model], [Latent Diffusion]


# Digital Watermark

 Digital watermarking is a method of embedding information into a digital signal in a way that is difficult to remove, but can be detected. This information can be used to identify the source of the digital signal, or to prevent unauthorized copying or tampering. Digital watermarks are often used to protect copyrights in digital media, such as images, audio, or video.

 More at:
  * [https://pub.towardsai.net/human-vs-gpt-methods-to-watermark-gpt-models-e23aefc63db8](https://pub.towardsai.net/human-vs-gpt-methods-to-watermark-gpt-models-e23aefc63db8)
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)
  * paper - [https://aclanthology.org/D19-1115.pdf](https://aclanthology.org/D19-1115.pdf)

 See also [D], [ChatGPT Model], [DALL-E Model], [GPT Model], [InstructGPT Model]


# Dimensionality Reduction

 Some problems may contain thousands or millions of features, which can be computationally costly to work with. Additionally, the program's ability to generalize may be reduced if some of the features capture noise or are irrelevant to the underlying relationship. Dimensionality reduction is the process of discovering the features that account for the greatest changes in the response variable. Dimensionality reduction can also be used to visualize data. It is easy to visualize a regression problem such as predicting the price of a home from its size; the size of the home can be plotted on the graph's x axis, and the price of the home can be plotted on the y axis. It is similarly easy to visualize the housing price regression problem when a second feature is added; the number of bathrooms in the house could be plotted on the z axis, for instance. A problem with thousands of features, however, becomes impossible to visualize.

  As the name suggests, we use dimensionality reduction to remove the least important information (sometime redundant columns) from a dataset. In practice, I often see datasets with hundreds or even thousands of columns (also called features), so reducing the total number is vital. For instance, images can include thousands of pixels, not all of which matter to your analysis. Or when testing microchips within the manufacturing process, you might have thousands of measurements and tests applied to every chip, many of which provide redundant information. In these cases, you need dimensionality reduction algorithms to make the dataset manageable. The most popular dimensionality reduction method is Principal Component Analysis (PCA). Other methods exist such as the Linear Discriminant Analysis (LDA). For visualisation, data scientists use the t-SNE method.

  See also [D], [Autoencoder], [Decoder], [Encoder], [Feature], [Linear Discriminant Analysis], [Principal Component Analysis], [t-SNE], [UMAP]


# Discovery Phase

 Before you build the ML model, you need to understand the problem. You may be the expert in ML, but you may not be the expert in problem. Ask questions to the domain experts. The more questions you ask the more relevant your model will be. Here are sample questions for the amazon call centre outing (multi-class) problem (i.e. to which agent-skill should a call be routed next?) :

 ![]( {{site.assets}}/d/discovery_phase_questions.png ){: width="100%"}


# Discrete Action Space

 In [Reinforcement Learning], when the [Action Space] is a set of actions.

 ![]( {{site.assets}}/d/discrete_action_space.png ){: width="100%"}

 See also [D], ...

# Discrete Variable

 A variable that takes a (finite?) set of numerical value.

 See also [D], [Categorical Variable], [Continuous Variable], [Variable Type]


# Discriminative Classifier

 Discriminative Classifiers learn what the features in the input are most useful to distinguish between the various possible classes. So, if given images of dogs and cats, and all the dog images have a collar, the discriminative models will learn that having a collar means the image is of dog. An example of a discriminative classifier is logistic regression. Mathematically, it directly calculates the posterior probability `P(y|x)` or learn a direct map from input x to label y. So, these models try to learn the decision boundary for the model.

 See also [D], [Artificial Neural Network], [Conditional Random Fields], [K-Nearest Neighbor], [Logistic Regression], [Scalar Vector Machine]


# Discriminator

 Answer the question is this a real Monet? picture? aka a bullshit detector! :-) Gives continuous feedback. For example: do you like this music, and as music is playing feedback is applied continuously. Works with another neural network, the generator, that generates the music/image and learn from the discriminator's feedback! How does the discriminator perform classification? Solution: The discriminator gets a probability score after convolutions and hence the discriminator chooses the decision based on the probability.  The goal of the discriminator is to provide feedback to the generator about how realistic the generated outputs (e.g. piano rolls) are, so that the generator can learn to produce more realistic data. The discriminator provides this feedback by outputting a scalar value that represents how “real” or “fake” a piano roll is. Since the discriminator tries to classify data as “real” or “fake”, it is not very different from commonly used binary classifiers. We use a simple architecture for the critic, composed of four convolutional layers and a dense layer at the end.

 ![]( {{site.assets}}/d/discriminator.png ){: width="100%"}

  * This feedback from the discriminator is used by the generator to update its weights.
  * As the generator gets better at creating music accompaniments, it begins fooling the discriminator. So, the discriminator needs to be retrained as well.
  * Beginning with the discriminator on the first iteration, we alternate between training these two networks until we reach some stop condition (ex: the algorithm has seen the entire dataset a certain number of times).

 See also [D], [Dense Layer], [Discriminator Loss], [Generative Adversarial Network], [Generator], [Loss Function], [Update Ratio]


# Discriminator Loss

 See also [D], [Generator Loss], [Loss Function], [Loss Graph]


# Disentangled Variational Autoencoder

# Beta-VAE

 Variational autoencoder where weights in the latent space are meaningful, e.g. rotation of the head in a portrait representation.

 {% pdf "{{site.assets}}/d/disentangled_variational_autoencoder_paper.pdf" %}

 Beware:
  * Beta too small - variables are disentangled, but maybe overfitting training set?
  * beta too big - variables are not disentangled enough

 More at:
  * [https://youtu.be/9zKuYvjFFS8?t=555](https://youtu.be/9zKuYvjFFS8?t=555)

 See also [D], [Kullback-Liebler Divergence], [Latent Perturbation], [Variational Autoencoder]


# DistilBert Model

 A smaller, but faster version of the BERT model.

 See also [D], [BERT Model]


# Distribution

 See also [D], [Cumulative Distribution Function], [Sample]


# Distributed Training

 See also [D], [Apache Spark], [TensorFlow ML Framework]


# Domain-Specific Model

 See also [D], [Supervised Fine-Tuning], [Model]


# Dot Product

 This is the same as doing a dot product and you can think of a dot product of two vectors as a measure of how similar they are.
 The dot product of two vectors has two definitions. Algebraically the dot product of two vectors is equal to the sum of the products of the individual components of the two vectors.

 ```
→ →
a.b = a1.b1 + a2.b2 + a3.b3     # Dot product of 2 vectors: A is 1-row/3-column with B is 3-row/1-column
 ```
 
 Geometrically the dot product of two vectors is the product of the magnitude of the vectors and the cosine of the angle between the two vectors.

 ```
→ →    →   →                                →
a.b = |a|.|b|. cos (θ)          # Where abs(a) = sqrt(a1^2 + a2^2 + a3^3) and theta angle between 2 vectors
 ```

 The resultant of the dot product of vectors is a scalar value.

 ![]( {{site.assets}}/d/dot_product.png ){: width="100%"}

 More at:
  * [https://www.cuemath.com/algebra/dot-product/](https://www.cuemath.com/algebra/dot-product/)

 See also [D], [Vector]


# Downstream Task

 See also [D], [Finetuning], [Supervised Learning], [Upstream Task]


# Drop Out

 Remove nodes from the Neural Network to prevent over-fitting.

 See also [D], [Hyperparameter]

