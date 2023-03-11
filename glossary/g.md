---
title: G
permalink: /g/

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

# Galactica Large Language Model

# Galactica LLM

 Galactica is “a large language model that can store, combine and reason about scientific knowledge,” according to a paper published by Meta AI. It is a transformer model that has been trained on a carefully curated dataset of 48 million papers, textbooks and lecture notes, millions of compounds and proteins, scientific websites, encyclopedias, and more. Galactica was supposed to help scientists navigate the ton of published scientific information. Its developers presented it as being able to find citations, summarize academic literature, solve math problems, and perform other tasks that help scientists in research and writing papers.

 {% pdf "{{site.assets}}/g/galactica_paper.pdf" %}

 More at:
  * what happened to galactica? - [https://www.louisbouchard.ai/galactica/](https://www.louisbouchard.ai/galactica/)
  * take aways - [https://bdtechtalks.com/2022/11/21/meta-ai-galactica](https://bdtechtalks.com/2022/11/21/meta-ai-galactica)
  * site - [http://galactica.org](http://galactica.org)

 See also [G], [Large Language Model], 


# Gated Recurrent Unit Cell

# GRU Cell

 Cell or module that can be used in the RNN chain of a Long Short Term Memory, or LSTM Network. A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by Cho, et al. (2014). It combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models and therefore less compute intensive. This cell has been growing increasingly popular.

 ![]( {{site.assets}}/g/gated_recurrent_unit_cell2.png ){: width="100%"}

 ![]( {{site.assets}}/g/gated_recurrent_unit_cell.png ){: width="100%"}

 More at:
  * [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

 See also [G], [Long Short Term Memory Network], 

# Gaussian Distribution

 The Gaussian distribution, normal distribution, or bell curve, is a probability distribution which accurately models a large number of phenomena in the world. Intuitively, it is the mathematical representation of the general truth that many measurable quantities, when taking in aggregate tend to be of the similar values with only a few outliers which is to say that many phenomena follow the central limit theorem.

 ![]( {{site.assets}}/g/gaussian_distribution.png ){: width="100%"}

 See also [G], [Central Limit Theorem], [Gaussian Process]

# Gaussian Process

 See also [G], [Random Forest], [Tree Parzen Estimators]

# Generative Adversarial Network

# GAN

 So why do we want a generative model? Well, it’s in the name! We wish to generate something an image, music, something! But what do we wish to generate? `Typically, we wish to generate data` (I know, not very specific). More than that though, it is likely that we wish to generate data that is never before seen, yet still fits into some data distribution (i.e. some pre-defined dataset that we have already set aside and that was used to build a discriminator). GANs, a generative AI technique, pit 2 networks against each other to generate new content. The algorithm consists of two competing networks: a generator and a discriminator. A generator is a convolutional neural network (CNN) that learns to create new data resembling the source data it was trained on. The discriminator is another convolutional neural network (CNN) that is trained to differentiate between real and synthetic data. The generator and the discriminator are trained in alternating cycles such that the generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.

 ![]( {{site.assets}}/g/vanilla_gan.png ){: width="100%"}

 There are different types of GAN, including:
  * Vanilla GAN
  * Cycle GAN
  * Conditional GAN
  * Deep Convoluted GAN
  * Style GAN
  * Super Resolution GAN (SRGAN)

 {% pdf "{{site.assets}}/g/generative_adversarial_networks_paper.pdf" %}

 More at:
  * [http://hunterheidenreich.com/blog/what-is-a-gan/](http://hunterheidenreich.com/blog/what-is-a-gan/)

 See also [G], [AR-CNN], [Convolutional Neural Network], [Conditional GAN], [Cycle GAN], [DeepComposer], [Discriminator], [Generative Model], [Generator]

# Generative Classifier

 Generative Classifiers tries to model class, i.e., what are the features of the class. In short, it models how a particular class would generate input data. When a new observation is given to these classifiers, it tries to predict which class would have most likely generated the given observation. Such methods try to learn about the environment. An example of such classifiers is Naive Bayes. Mathematically, generative models try to learn the joint probability distribution, `p(x,y)`, of the inputs x and label y, and make their prediction using Bayes rule to calculate the conditional probability, `p(y|x)`, and then picking a most likely label. Thus, it tries to learn the actual distribution of the class.

 ![]( {{site.assets}}/g/generative_classifier.png ){: width="100%"}

 See also [G], [Bayesian Network], [Hidden Markov Model], [Markov Random Field], [Naive Bayes]

# Generative Model

 AI models that generate/create content. Examples of Generative AI techniques include:
  * Generative Adversarial Networks (GAN) 
  * Variational autoencoders (VAEs) = Hidden state is represented by a distribution, which is then sampled and decoded (Q: what is mean and variance?)
  * Transformers 
   * Decoders (with masked attention)

 ![]( {{site.assets}}/g/generative_model_1.png ){: width="100%"}

 ![]( {{site.assets}}/g/generative_model_2.png ){: width="100%"}

 More at :
  * Generative Modeling by Estimating Gradients of the Data Distribution - [https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)

 See also [G], [Decoder], [Diffusion Model], [Flow-Based Model], [Generative Adversarial Network], [Masked Attention], [Transformer Model], [Variational Autoencoder]
 


# Generative Pre-Trained Transformer Model
# GPT Model
# GPT3

 Before GPT-3 there was no general language model that could perform well on an array of NLP tasks. Language models were designed to perform one specific NLP task, such as text generation, summarization, or classification, using existing algorithms and architectures. GPT-3 has extraordinary capabilities as a general language model. GPT-3 is pre-trained on a corpus of text from five datasets: Common Crawl, !WebText2, Books1, Books2, and Wikipedia. 
  * By default, GPT2 remembers the last 1024 words. That the max? length of the left-side context?
  * GPT-3 possesses 175 billion weights connecting the equivalent of 8.3 million neurons arranged 384 layers deep.

 {% pdf "{{site.assets}}/g/gpt3_model_paper.pdf" %}

 More at:
   * GPT-1 paper - [https://paperswithcode.com/paper/improving-language-understanding-by](https://paperswithcode.com/paper/improving-language-understanding-by)
   * GPT-2 paper - [https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (also attached)
   * GPT fine-tuning - [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
   * gpt vs chatgpt vs instructgpt - [https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5](https://medium.com/@colin.fraser/chatgpt-automatic-expensive-bs-at-scale-a113692b13d5)

 See also [G], [ChatGPT Model], [Digital Watermark], [Fine-Tuning], [InstructGPT Model], [Natural Language Processing], [Pre-Trained Model]

# Generator

 A neural network that generates music, image, something and is getting feedback from a Discriminator neural network. How does the generator generate images? Solution : The generator takes noise from latent space and further based on test and trial feedback , the generator keeps improvising on images. After a certain times of trial and error , the generator starts producing accurate images of a certain class which are quite difficult to differentiate from real images.

 See also [G], [DeepComposer], [Discriminator], [Generative Adversarial Network], [Generator Loss], [Latent Space]

# Generator Loss

 Typically when training any sort of model, it is a standard practice to monitor the value of the loss function throughout the duration of the training. The discriminator loss has been found to correlate well with sample quality. You should expect the discriminator loss to converge to zero and the generator loss to converge to some number which need not be zero. When the loss function plateaus, it is an indicator that the model is no longer learning. At this point, you can stop training the model. You can view these loss function graphs in the AWS DeepComposer console:

 ![]( {{site.assets}}/g/generator_loss.png ){: width="100%"}

 After 400 epochs of training, discriminator loss approaches near zero and the generator converges to a steady-state value. Loss is useful as an evaluation metric since the model will not improve as much or stop improving entirely when the loss plateaus.

 See also [G], [Discriminator Loss], [Loss Function], [Loss Graph]

# Genetic Programming

 Genetic programming is a technique to create algorithms that can program themselves by simulating biological breeding and Darwinian evolution. Instead of programming a model that can solve a particular problem, genetic programming only provides a general objective and lets the model figure out the details itself. The basic approach is to let the machine automatically test various simple evolutionary algorithms and then “breed” the most successful programs in new generations.

# GINI Impurity Index

 A metrics to measure how diverse the data is in a dataset.
  * The more diverse the dataset is, the closer the GINI index is to 1 (but never equal to one)
  * The GINI index = 1 in the impossible case where all the elements in the dataset are different and the dataset is na infinite number of sample
  * The GINI index is 0 if all the samples in the dataset are the same (same label)

 {% youtube "https://www.youtube.com/watch?v=u4IxOk2ijSs" %}

 Questions:
   * Why is this important?

 See also [G], [Dataset], [Forest Of Stumps], [Weighted Gini Impurity Index]

# GLIDE Model

 What adds the text conditioning to the diffusion model!?

 {% youtube "https://www.youtube.com/watch?v=lvv4N2nf-HU" %}

 {% pdf "{{site.assets}}/g/glide_model_paper.pdf" %}

 More at :
  * how does DALL-E work? - [https://www.assemblyai.com/blog/how-dall-e-2-actually-works/](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)
  * blog - [https://syncedreview.com/2021/12/24/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-173/](https://syncedreview.com/2021/12/24/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-173/)
  * paper - [https://arxiv.org/abs/2112.10741](https://arxiv.org/abs/2112.10741)
  * code - [https://github.com/openai/glide-text2im](https://github.com/openai/glide-text2im)
  * colab notebooks - [https://github.com/openai/glide-text2im/tree/main/notebooks](https://github.com/openai/glide-text2im/tree/main/notebooks)

 See also [G], [DALL-E Model]

# GloVe

 See also [G], [Word Embedding Space]

# Gluon

 This is ...

# Google Company

 Models developed by the company
   * BERT Model
   * Chinchila Model
   * Gshard Model
   * LaMBDA Model
   * Minerva Model
   * MUM Model
   * Muse Model
   * MusicLM Model
   * PaLM Model
   * Pathways Model Architecture
   * Sparrow (by !DeepMind)
   * T5 Model
   * TPU
   * Transformer
 Companies
   * DeepMind

 See also [G], [BERT Model], [DeepMind Company], [Gshard Model], [LaMBDA Model], [Minerva Model], [MUM Model], [Muse Model], [MusicLM Model], [PaLM Model], [Pathways Model Architecture], [Sparrow Model], [Switch Transformer], [T5 Model], [Tensor Processing Unit], [Transformer Model]

# Gopher Model

 NLP model developed by deepmind.

 More at :
  * [https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval)

 See also [G], [Deepmind], [GPT Model]

# GPU Instance

 P2 and P3 instances on AWS.

 See also [G], [Graphical Processing Unit]

# Gradient

 A gradient is the direction and magnitude calculated during the training of a neural network it is used to teach the network weights in the right direction by the right amount.

 See also [G], [Gradient Descent]

# Gradient Boosting

 Use gradient descent to minimize the loss function.

 See also [G], [Boosting], [Ensemble Method], [Gradient Bagging], [Weak Learner]

# Gradient Clipping

 A technique to prevent exploding gradients in very deep networks, usually in recurrent neural networks. Gradient Clipping is a method where the error derivative is changed or clipped to a threshold during backward propagation through the network, and using the clipped gradients to update the weights.
 Gradient Clipping is implemented in two variants:
  * Clipping-by-value
  * Clipping-by-norm

 More at: 
  * [https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48)
  * gradient clipping accelerate learning? - [https://openreview.net/pdf?id=BJgnXpVYwS](https://openreview.net/pdf?id=BJgnXpVYwS)

 See also [G], [Exploding Gradient Problem], [Gradient Descent], [Recurring Neural Network]

# Gradient Descent

 One of the shining successes in machine learning is the gradient descent algorithm (and its modified counterpart, stochastic gradient descent). Gradient descent is an iterative method for finding the minimum of a function. In machine learning, that function is typically the loss (or cost) function. "Loss" is simply some metric that quantifies the cost of wrong predictions. Gradient descent calculates the loss achieved by a model with a given set of parameters, and then alters those parameters to reduce the loss. It repeats this process until that loss can't substantially be reduced further. The final set of parameters that minimize the loss now define your fitted model. 

 {% youtube "https://www.youtube.com/watch?v=OkmNXy7er84" %}

 ![]( {{site.assets}}/g/gradient_descent.png ){: width="100%"}

 `Using gradient descent, you can find the regression line that best fit the sample using a loss function` (distance of sample from line). To do that, you can start with any line and calculate the loss function, as you move the parameter, the loss function become smaller and smaller indicating in which direction the parameter should go.

 :warning: On huge datasets, There are downsides of the gradient descent algorithm. Indeed for huge datasets, we need to take a huge amount of computation we make for each iteration of the algorithm. To alleviate the problem, possible solutions are:
  * batch data for lighter partial processing (= standard gradient descent) :warning: compute intensive for large datasets
  * mini-batch ... = uses all the data in batches (= best of all proposed alternatives?)
  * data sampling or stochastic gradient descent. (= lower compute required since we use only a few sample, but faster iteration with slower convergence st each step, but faster overall?)

 See also [G], [Activation Function], [Batch Gradient Descent], [Gradient Perturbation], [Learning Rate], [Loss Function], [Mini-Batch Gradient Descent], [Parameter], [Prediction Error], [Stochastic Gradient Descent]

# Gradient Perturbation

 Works for neural network only or where gradient descent is used. Better than output perturbation in that the noise is built in the model, so you can share the model at will ;-) Gradient perturbation, widely used for differentially private optimization, injects noise at every iterative update to guarantee differential privacy. Noise is included in the gradient descent!

 More at:
  * [https://www.ijcai.org/Proceedings/2020/431}(https://www.ijcai.org/Proceedings/2020/431)
  * deep learning with differential privacy paper - [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)
  * paper - [https://www.microsoft.com/en-us/research/publication/gradient-perturbation-is-underrated-for-differentially-private-convex-optimization/](https://www.microsoft.com/en-us/research/publication/gradient-perturbation-is-underrated-for-differentially-private-convex-optimization/)

 See also [G], [Differential Privacy], [Gradient Clipping], [Output Perturbation]

# Graph Convolutional Network

# GCN

 More at 
  * [https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780](https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780)

 See also [G], ...

# Graph Neural Network

# GNN

 Use graph relationship to ... GNNs are used to train predictive models on datasets such as:
  * Social networks, where graphs show connections between related people,
  * Recommender systems, where graphs show interactions between customers and items,
  * Chemical analysis, where compounds are modeled as graphs of atoms and bonds,
  * Cybersecurity, where graphs describe connections between source and destination IP addresses,
  * And more!

 Ex: Most of the time, these datasets are extremely large and only partially labeled. Consider a fraud detection scenario where we would try to predict the likelihood that an individual is a fraudulent actor by analyzing his connections to known fraudsters. This problem could be defined as a semi-supervised learning task, where only a fraction of graph nodes would be labeled (‘fraudster’ or ‘legitimate’). This should be a better solution than trying to build a large hand-labeled dataset, and “linearizing” it to apply traditional machine learning algorithms.

 More at:
  * [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)

 See also [G], [Entity Extraction], [GNN Edge-Level Task], [GNN Graph-Level Task], [GNN Node-Level Task], [Insufficient Data Algorithm], [Knowledge Graph], [Relation Extraction], [Scene Graph]

# Graph Neural Network Edge-Level Task

# GNN Edge-Level Task

 One example of edge-level inference is in image scene understanding. Beyond identifying objects in an image, deep learning models can be used to predict the relationship between them. We can phrase this as an edge-level classification: given nodes that represent the objects in the image, we wish to predict which of these nodes share an edge or what the value of that edge is. If we wish to discover connections between entities, we could consider the graph fully connected and based on their predicted value prune edges to arrive at a sparse graph.

 ![]( {{site.assets}}/g/graph_neural_network_edge_level_task.png ){: width="100%"}

 In (b), above, the original image (a) has been segmented into five entities: each of the fighters, the referee, the audience and the mat. (C) shows the relationships between these entities.

 ![]( {{site.assets}}/g/graph_neural_network_edge_level_task2.png ){: width="100%"}

 See also [G], [Graph Neural Network]

# Graph Neural Network Graph Level Task

# GNN Graph-Level Task

 In a graph-level task, our goal is to predict the property of an entire graph. For example, for a molecule represented as a graph, we might want to predict what the molecule smells like, or whether it will bind to a receptor implicated in a disease.

 ![]( {{site.assets}}/g/graph_neural_network_graph_level_task.png ){: width="100%"}

 This is analogous to image classification problems with MNIST and CIFAR, where we want to associate a label to an entire image. With text, a similar problem is sentiment analysis where we want to identify the mood or emotion of an entire sentence at once.

 See also [G], [Graph Neural Network]

# Graph Neural Network Node-Level Task

# GNN Node-Level Task

 Node-level tasks are concerned with predicting the identity or role of each node within a graph. A classic example of a node-level prediction problem is Zach’s karate club. The dataset is a single social network graph made up of individuals that have sworn allegiance to one of two karate clubs after a political rift. As the story goes, a feud between Mr. Hi (Instructor) and John H (Administrator) creates a schism in the karate club. The nodes represent individual karate practitioners, and the edges represent interactions between these members outside of karate. The prediction problem is to classify whether a given member becomes loyal to either Mr. Hi or John H, after the feud. In this case, distance between a node to either the Instructor or Administrator is highly correlated to this label.

 ![]( {{site.assets}}/g/graph_neural_network_node_level_task.png ){: width="100%"}

 On the left we have the initial conditions of the problem, on the right we have a possible solution, where each node has been classified based on the alliance. The dataset can be used in other graph problems like unsupervised learning.

 Following the image analogy, node-level prediction problems are analogous to image segmentation, where we are trying to label the role of each pixel in an image. With text, a similar task would be predicting the parts-of-speech of each word in a sentence (e.g. noun, verb, adverb, etc).

 See also [G], [Graph Neural Network]

# Graphical Processing Unit

# GPU

 To accelerate processing of data.

 See also [G], [CPU], [Cuda Core], [Hyperparameter Optimization], [TPU]

# Grid Search

 Combinatorial growth.

 See also [G], [Hyperparameter Optimization], [Random Search]

# Gshard Model

  * [https://arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668)

 See also [G], [Google Company], [Sparse Activation]

# GSMK Dataset

 GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning.

 More at
  * paper - [https://paperswithcode.com/paper/training-verifiers-to-solve-math-word](https://paperswithcode.com/paper/training-verifiers-to-solve-math-word)
  * site - [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
  * dataset - [https://paperswithcode.com/dataset/gsm8k](https://paperswithcode.com/dataset/gsm8k)

 See also [G], [Dataset], [OpenAI Company], [PaLM Model]
