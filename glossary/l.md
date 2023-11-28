---
title: L
permalink: /l/

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


# Label

 ~ think of your label as your model teacher!

 Name of a prediction in a supervised models. Correspond to a target attribute in unsupervised learning. Example of label: the agent-skill needed to result the customer's call.

 See also [L], [Data Point], [Labeling Function], [Supervised Learning], [Target Attribute]


# Labeling Function

 Use one or more function to label a sample. If you have more than one labeling function, use the majority rule, i.e. label the sample with the sample that has the maximum probability. :warning: A label is an enum, not a probability. Used by snorkel.

 See also [L], [Label], [Snorkel Program]


# Labeling Service

 Mechanical turk, crowd flower, instaML LOOP. Do you have the proper label? Have several people label the same image/entry and used the Dawid-Skene or majority vote algorithm!

 See also [L], [Dawid-Skene Algorithm], [Majority Vote Algorithm], [Unlabelled Data Algorithm]


# Labor Market Impact

 We investigate the potential implications of [large language models (LLMs)][Large Language Model], such as [Generative Pre-trained Transformers (GPTs)][GPT Model], on the U.S. labor market, focusing on the increased capabilities arising from LLM-powered software compared to LLMs on their own. Using a new rubric, we assess occupations based on their alignment with LLM capabilities, integrating both human expertise and GPT-4 classifications. 

 **Our findings reveal that around 80% of the U.S. workforce could have at least 10% of their work tasks affected by the introduction of LLMs, while approximately 19% of workers may see at least 50% of their tasks impacted**.

 We do not make predictions about the development or adoption timeline of such LLMs. The projected effects span all wage levels, with higher-income jobs potentially facing greater exposure to LLM capabilities and LLM-powered software. Significantly, these impacts are not restricted to industries with higher recent productivity growth. Our analysis suggests that, with access to an LLM, about 15% of all worker tasks in the US could be completed significantly faster at the same level of quality. When incorporating software and tooling built on top of LLMs, this share increases to between 47 and 56% of all tasks. This finding implies that LLM-powered software will have a substantial effect on scaling the economic impacts of the underlying models. We conclude that LLMs such as GPTs exhibit traits of general-purpose technologies, indicating that they could have considerable economic, social, and policy implications.

 {% youtube "https://www.youtube.com/watch?v=ooqYC781HGE" %}

 {% pdf "https://arxiv.org/pdf/2303.10130.pdf" %}

 More at:
  * [https://openai.com/research/gpts-are-gpts](https://openai.com/research/gpts-are-gpts)

 See also [L], ...


# LangChain Python Module

 LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an API, but will also:
  * Be data-aware: connect a language model to other sources of data
  * Be agentic: allow a language model to interact with its environment

 The LangChain framework is designed with the above principles in mind.

 ```
# Proprietary LLM from e.g. OpenAI
# pip install openai
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")

# Alternatively, open-source LLM hosted on Hugging Face
# pip install huggingface_hub
from langchain import HuggingFaceHub
llm = HuggingFaceHub(repo_id = "google/flan-t5-xl")

# The LLM takes a prompt as an input and outputs a completion
prompt = "Alice has a parrot. What animal is Alice's pet?"
completion = llm(prompt)
 ```

 ![]( {{site.assets}}/l/langchain_python_module_value_proposition.png ){: width="100%"}
 ![]( {{site.assets}}/l/langchain_python_module_components.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=nE2skSRWTTs" %}

 More at:
  * announcement - [https://www.pinecone.io/learn/langchain-intro/](https://www.pinecone.io/learn/langchain-intro/)
  * docs - [https://python.langchain.com/en/latest/index.html](https://python.langchain.com/en/latest/index.html)
  * JS docs - [https://js.langchain.com/docs/](https://js.langchain.com/docs/)
  * tutorials
   * book - [https://www.pinecone.io/learn/langchain/](https://www.pinecone.io/learn/langchain/)
   * notebooks - [https://github.com/pinecone-io/examples/tree/master/generation/langchain/handbook](https://github.com/pinecone-io/examples/tree/master/generation/langchain/handbook)
  * articles
   * [https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c](https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c)
  * colabs
   * intro - [https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/langchain/handbook/00-langchain-intro.ipynb](https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/langchain/handbook/00-langchain-intro.ipynb)

 See also [L], [Vector Database]


# Language Model

 See also [L], [Language Modeling], [Large Language Model]


# Language Model for Discussion Applications Model 
# LaMDA Model

 Built by [Google]

 Beware, cannot use GPU for inference. ??? <== ????

 {% youtube "https://www.youtube.com/watch?v=7BvbgUNT2gI" %}

 {% youtube "https://www.youtube.com/watch?v=7BvbgUNT2gI" %}

 More at:
  * blog - [https://blog.google/technology/ai/lamda/](https://blog.google/technology/ai/lamda/)

 See also [L], ...


# Language Modeling

 Language modeling is the task of assigning a probability to a sequence of words in a text in a specific language. Simple language models can look at a word and predict the next word (or words) most likely to follow it, based on statistical analysis of existing text sequences. To create a language model that successfully predicts word sequences, you need to train it on large sets of data. Language models are a key component in natural language processing applications. You can think of them as statistical prediction machines, where you give text as input and get a prediction as the output. You’re probably familiar with this from the auto-complete feature on your smartphone. For instance, if you type “good,” auto-complete might suggest “morning” or “luck.”

 See also [L], [Language Model], [Large Language Model], [Natural Language Processing]


# Language Parsing

 ~ figuring out which group of words go together (as “phrases”) and which words are the subject or object of a verb. The NLP parser separates a series of text into smaller pieces based on the grammar rules. If a sentence that cannot be parsed may have grammatical errors.

 See also [L], [Benchmark]


# Large Language and Vision Assistant Model
# LLaVa Model

 An extension to the [LLaMA Model] to allow it to be multimodal or see.

 Lava is a recently released multimodal model called Large Language and Vision Assistant. It can run multimodal tasks across both image and text inputs. Lava has shown promising performance in understanding and reasoning about images, generating HTML websites from wireframe sketches, and generating stories based on complex images. Its ability to process both visual and textual information sets it apart from traditional language models.

 {% youtube "https://www.youtube.com/watch?v=RxBSmbdJ1I8" %}

 More at:
  * demo - [https://llava.hliu.cc/](https://llava.hliu.cc/)
  * code - [https://github.com/haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
  * project site - [https://llava-vl.github.io/](https://llava-vl.github.io/)

 See also [L], ...

 
# Large Language Model
# LLM

 Large Language Models are [Language Model] with not millions, but billions of parameters/weights. The term "large" in LLM refers to the fact that these models are designed to handle large amounts of data, both in terms of the size of the text corpus used to train them and in terms of the amount of text they can generate or process at once.

 In 2023, aftr the release of [ChatGPT][ChatGPT Model], LLMs started having a huge [impact on the labor force][Labor Market Impact]

 These models typically utilize deep learning techniques and are trained on massive amounts of text data, such as books, articles, and web pages, in order to learn the patterns and structure of language.

 Examples of Large Language Models include
  * [GPT-3, GPT-2][GPT Model],
  * [BERT][BERT Model],
  * and [T5][T5 Model], among others.

 These models have been used for a variety of tasks, such as [language translation][Machine Translation], text generation, [question answering], and [sentiment analysis], and have demonstrated impressive performance on many [benchmarks][Benchmark] in [natural language understanding] and generation.

 {% youtube "https://www.youtube.com/watch?v=StLtMcsbQes" %}

 See also [L], [Language Modeling], [Model Compression], [Scaling Law]


# Large Language Model Operations
# LLMOps

  * where you validate improvements over baseline

 See also [L], ...


# Large Language Model Self-Correction Reasoning
# LLM Self-Correction Reasoning

 ![]( {{site.assets}}/l/large_language_model_self_correction_reasoning.png ){: width="100%"}

 {% pdf "https://arxiv.org/pdf/2308.03188.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2308.03188](https://arxiv.org/abs/2308.03188)
  * github - [https://github.com/teacherpeterpan/self-correction-llm-papers](https://github.com/teacherpeterpan/self-correction-llm-papers)
  * [https://bdtechtalks.com/2023/10/09/llm-self-correction-reasoning-failures/](https://bdtechtalks.com/2023/10/09/llm-self-correction-reasoning-failures/)

 See also [L], ....


# Lasso Regression

 Used in [Regularization].

 More at:
  * [https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/)

 See also [L], ...


# Latent Diffusion Model
# LDM

 ![]( {{site.assets}}/l/latent_diffusion_model.png ){: width="100%"}

 The overall model will look like this:
  * you will have your initial image here X, and encode it into an information-dense space called the latent space, Z. This is very similar to a GAN where you will use an encoder model to take the image and extract the most relevant information about it in a sub-space, which you can see as a downsampling task. Reducing its size while keeping as much information as possible.
  * You are now in the latent space with your condensed input. You then do the same thing with your conditioning inputs, either text, images, or anything else,
  * and merge them with your current image representation. WE condition LDMs either via concatenation or by a more general cross-attention mechanism. This attention mechanism will learn the best way to combine the input and conditioning inputs in this latent space. Adding attention, a transformer feature, to diffusion models. These merged inputs are now your initial noise for the diffusion process. Then, you have the same diffusion model I covered in my Imagen video but still in this sub-space.
  * Finally, you reconstruct the image using a decoder which you can see as the reverse step of your initial encoder. Taking this modified and de-noised input in the latent space to construct a final high-resolution image, basically upsampling your result.
 And voilà! This is how you can use diffusion models for a wide variety of tasks like super-resolution, inpainting, and even text-to-image with the recent stable diffusion open-sourced model through the conditioning process while being much more efficient and allowing you to run them on your GPUs instead of requiring hundreds of them. 

 {% pdf "{{site.assets}}/l/latent_diffusion_model_paper.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=RGBNdD3Wn-g" %}

 More at :
  * [https://pub.towardsai.net/latent-diffusion-models-the-architecture-behind-stable-diffusion-434ba7d91108](https://pub.towardsai.net/latent-diffusion-models-the-architecture-behind-stable-diffusion-434ba7d91108)
  * [https://www.louisbouchard.ai/latent-diffusion-models/](https://www.louisbouchard.ai/latent-diffusion-models/)
  * [code - https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)

 See also [L], [Conditioning], [Cross-Attention], [Diffusion Model], [Diffusion Process], [Image Decoder], [Image Encoder], [Latent Space], [Pixel Space], [U-Net Architecture]


# Latent Dirichlet Allocation
# LDA

 Used as a topic modeling technique that is it can classify text in a document to a particular topic. It uses Dirichlet distribution to find topics for each document model and words for each topic model. Johann Peter Gustav Lejeune Dirichlet was a German mathematician in the 1800s who contributed widely to the field of modern mathematics. There is a probability distribution named after him ‘Dirichlet Distribution’ which is the basis of Latent Dirichlet Allocation (--LDA--).

 More at:
  * ...

 See also [L], ...


# Latent Perturbation

 Used to find out what the latent variable are/mean in a latent variable model. :warning: The model learns by itself that those are important variables based on the provided training sample. :warning: The loss function defines what is learned and HOW it learns it! Latent perturbation is useful to see how entangled or disentangled latent variables are.

 See also [L], [Disentangled Variational Autoencoder], [Latent Variable], [Latent Variable Model]


# Latent Space

 `~ A compressed/downsampled space that contains as much information as possible space`. Formally, a latent space is defined as an abstract multi-dimensional space that encodes a meaningful internal representation of externally observed events. Samples that are similar in the external world are positioned close to each other in the latent space. To better understand the concept, let’s think about how humans perceive the world. We are able to understand a broad range of topics by encoding each observed event in a compressed representation in our brain. For example, we don’t keep in mind every detail of the appearance of a dog to be able to recognize a dog in the street. As we can see in the illustration below, we keep an internal representation of the general appearance of a dog:

 ![]( {{site.assets}}/l/latent_space_in_mind.png ){: width="100%"}

 In a similar way, the latent space tries to provide a compressed understanding of the world to a computer through a spatial representation.
 Deep learning has revolutionized many aspects of our life with applications ranging from self-driving cars to predicting serious diseases. Its main goal is to transform the raw data (such as the pixel values of an image) into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input. So, we realize that deep learning and latent space are strongly related concepts since the internal representations of the former constitute the latter. As we can see below, a deep learning model takes as input raw data and outputs discriminative features that lie in a low-dimensional space referred to as latent space. These features are then used to solve various tasks like classification, regression, or reconstruction:

 ![]( {{site.assets}}/l/latent_space.png ){: width="100%"}

 To better understand the importance of latent space in deep learning, we should think of the following question: Why do we have to encode the raw data in a low-dimensional l atent space before classification, regression, or reconstruction?
 The answer is data compression. Specifically, in cases where our input data are high-dimensional, it is impossible to learn important information directly from the raw data. 

 More at:
  * [https://ai.stackexchange.com/questions/11285/what-is-the-difference-between-latent-and-embedding-spaces](https://ai.stackexchange.com/questions/11285/what-is-the-difference-between-latent-and-embedding-spaces)

 See also [L], [Convolutional Neural Network], [Encoder]][Latent Variable], [Latent Variable Model], [Latent Vector], [Pixel Space], [Representation Space], [Semantic Space], [Word Embedding Space]


# Latent Space Compression

 {% youtube "https://www.youtube.com/watch?v=NqmMnjJ6GEg" %}

 See also [L], [Encoder], [Latent Space]


# Latent Variable

 Myth of the cave = where observation are only a projection of other objects. Latent variables are not directly observable, but are the true explanatory factors (that are casting the shadows that we see !)

 ![]( {{site.assets}}/l/latent_variable.png ){: width="100%"}

 See also [L], [Latent Space], [Latent Variable Model]


# Latent Variable Model

 See also [L], [Autoencoder], [Generative Adversarial Network], [Latent Space], [Latent Variable], [Variational Autoencoder]


# Latent Vector

 The input of a GAN acts as a latent vector since it encodes the output image \mathbf{G(z)} in a low-dimensional vector \mathbf{z}. To verify this, we can see how interpolation works in the latent space since we can handle specific attributes of the image by linearly modifying the latent vector. In the image below, we can see how we can handle the pose of a face by changing the latent vector of the GAN that generates it: 

 ![]( {{site.assets}}/l/latent_vector.png ){: width="100%"}

 See also [L], [Latent Space]


# Layer

 See also [L], [Hidden Layer], [Input Layer], [Output Layer]


# LeakyReLU Activation Function

 See also [L], [Activation Function], [Exploding Gradient Problem], [ReLU Activation Function], [Vanishing Gradient Problem]


# Learning Method

 All of those are or should be machine learning algorithm type! Here is a non-exhaustive list:
  * experience - learn from the past/data
  * [unsupervised learning] - try, fail, learn from failures ? Takes a long time / many iteration!
    * [association rule learning] -
  * [imitation learning] - clone behavior of experts <== good to get started, but do you understand?
  * [supervised learning] - with a teacher
  * [reinforcement learning] - reward-and-policy-based learning
  * [task-based learning] - focus on goal, use all of your skills to complete it and develop new ones (be motivated to find new skills)
  * [feedback-based learning] - get feedback from the crowd (experts and non-experts), select the feedback you want -- always try your best --> develop a persona
  * [transfer learning] - priors + I learned that concept before, no need to relearn
  * [weak-supervised learning] - augment the data (i.e. create data!) which has been labeled (supervised)
  * [semi-supervised learning] - label existing data based on data that has been labeled
  * [self-supervised learning] - acquire knowledge and skills through experiences and interactions without external feedback or instruction
  * [contrastive learning] - learning based on similarities and differences
  * [adaptive learning] - learning adapted to the learner's level and what has not yet been understood
  * [curriculum learning] - learning from simple to complex in order to learn faster and more efficiently.

 See also [L], [Machine Learning Type]


# Learning Process

 * Changing weights in an ANN using backpropagation

 See also [L], [Backpropagation]


# Learning Rate

 ~ controls how rapidly the model learns/changes

 :warning: Often symbolized by 'alpha'

 The learning rate `controls how rapidly the weights and biases of each network are updated during training`. A higher learning rate might allow the network to explore a wider set of model weights, but might pass over more optimal weights. Iterative learning: (1) observe difference between predicted answer, and correct answer. (2) Adjust the model a 'small amount' (at each pass /epoch) to make the prediction closer to the correct answer. Size of update at each iteration. Relative weight of new iteration vs old iterations?

 The learning rate is impacted differently function of the ML algorithm in use
 ```
new_value = expected_value + alpha * ( observed_error )
          = expected_value + alpha * ( observed_value - expected_value)
          = (1 - alpha) * expected_value + alpha * observed_value

with alpha = learning_rate
 ```
 In [reinforcement learning], more specifically in Q-learning, the learning rate is used as follow:

 ```
# Q-Learning

Q_new = (1 - alpha) * Q_old + alpha * Q_learned

# From state, go to next_state
# Q_old = value in the Q-table for the state-action pair
# Q_learned = computed value in the Q-table for the state-action pair given the latest action
            = R_t+1 + gamma * optimized_Q_value(next_state)               <== next state is known & next-state Q-values are known
            = R_t+1 + gamma * max( Q_current(next_state, action_i) )
  ```

 Beware:
  * the learning rate, alpha, is between 0 and 1
  * if alpha = 1  ==> immediately forget the past!
  * if alpha = 0  ==> oblivious to observation = no change!
  * A starting value can be between 0.01 and 0.1 which implies that updates with be between 1% and 10% of the observed error.

 See also [L], [Gradient Descent Algorithm], [Hyperparameter], [Loss Function], [Prior], [Transfer Learning]


# Learning Strategy

 See also [L], [Learning Method], [Learning Rate], [Learning Velocity]


# Learning Vector Quantization Algorithm
# LVQ Algorithm

 Clustering algorithm used in [unsupervised learning].

 {% youtube "https://www.youtube.com/watch?v=iq8aFkZo67o" %}

 More at:
  * ...

 See also [L], ...


# Learning Velocity

 How fast you learn to execute a task.

 See also [L], [Learning Rate], [Sample Strategy], [Sample Efficiency]


# Leave-One-Out Cross-Validation
# LOOCV

 A special case of [k-fold cross-validation] is the Leave-one-out cross-validation (LOOCV) method in which we set k=n (number of observations in the dataset). Only one training sample is used for testing during each iteration. This method is very useful when working with very small datasets.

 More at:
  * [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)

 See also [L], ...


# Lexical Search

 Word matching. Keyword search or exact phrase.

 Algorithms:
  * Rabin-Karp
  * Bayer-Moore
  * Knuth-Morris-Pratt

 ![]( {{site.assets}}/l/lexical_search_problem.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=au59-CEPegg" %}

 More at:
  * ...

 See also [L], [Semantic Search]


# LIDAR 

 See also [L], [Autonomous Vehicle]


# Light Gradient Boosting Machine
# LightGBM

 An [ensemble method].

 LightGBM, short for light gradient-boosting machine, is a free and open-source distributed gradient-boosting framework for machine learning, originally developed by Microsoft. It is based on [Decision tree algorithms][Decision Tree] and used for ranking, [classification] and other machine learning tasks. The development focus is on performance and scalability.

 {% pdf "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=R5FB1ZUejXM" %}

 More at:
  * docs - [https://lightgbm.readthedocs.io/en/latest/index.html](https://lightgbm.readthedocs.io/en/latest/index.html)
  * code - [https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
  * wikipedia - [https://en.wikipedia.org/wiki/LightGBM](https://en.wikipedia.org/wiki/LightGBM)

 See also [L], [Ensemble Method]


# Likelihood

 Another word for a probability in a discrete space/word/exercise

 See also [L], ...


# Linear Activation Function

 It is a simple straight-line [activation function] which is directly proportional to the input i.e. the weighted sum of neurons. It has the equation:

```
f(x) = kx
```

 where k is a constant.

 ![]( {{site.assets}}/l/linear_activation_function.png ){: width="100%"}

 More at:
  * ...

 See also [L], ...


# Linear Algebra

 Math where you do NOT have square, cubes, etc.

 More at:
  * [https://en.wikipedia.org/wiki/Linear_algebra](https://en.wikipedia.org/wiki/Linear_algebra)

 See also [L], ...


# Linear Autoencoder

 Let’s first suppose that both our encoder and decoder architectures have only one layer without non-linearity (linear autoencoder). Such encoder and decoder are then simple linear transformations that can be expressed as matrices. In such situation, we can see a clear link with PCA in the sense that, just like PCA does, we are looking for the best linear subspace (hidden state?) to project data on with as few information loss as possible when doing so. Encoding and decoding matrices obtained with PCA define naturally one of the solutions we would be satisfied to reach by gradient descent, but we should outline that this is not the only one.

 See also [L], [Autoencoder], [Principal Component Analysis]


# Linear Discriminant Analysis
# LDA

 Linear Discriminant Analysis(or LDA for short) was proposed by Ronald Fisher which is a Supervised Learning algorithm. It means that you must use both features and labels of data to reduce dimension while PCA only uses features. Another key point : the purpose of LDA is to find a new space in which reduced-dimension dataset is good for classification task. To meet this goal, LDA uses 2 metrics: Within-class variance and Between-class variance. The core idea is quite straightforward: finding vectors w which maximize the distance between mean vectors of 2 classes and minimize the variance within each class. A little bit explanation: within-class variance stands for scatter. The smaller this quantity, the lower data points scatter and vice versa. We want to classify classes, of course we have to maximize the distance between each class, that's why maximizing distance between mean vectors. However, we also need to take into account the scatter of data.The greater the within-class variance, the more data points of 2 classes overlap and it culminates in bad result for classification. Now you know why we need to minimize the scatter.

 More at:
  * [https://iq.opengenus.org/pca-vs-lda/](https://iq.opengenus.org/pca-vs-lda/)

 See also [L], [Dimensionality Reduction], [Principal Component Analysis]


# Linear Programming

 {% youtube "https://www.youtube.com/watch?v=Bzzqx1F23a8" %}

 More at:
  * ...

 See also [L], [Objective Function]


# Linear Regression

 Find an equation. Best fit. Ex: https://www.desmos.com/calculator/fmhotfn3qm.

 Not how long it will take for my car to stop given my speed (linear regression), but whether I am going to hit the tree or not (logistic regression).

 ![]( {{site.assets}}/l/linear_regression.png ){: width="100%"}

 Sample code:

 ```
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print regr.predict)X_test)
print regr.score(X_test, y_test)
 ```

 More at :
  * simulation [https://setosa.io/ev/ordinary-least-squares-regression/](https://setosa.io/ev/ordinary-least-squares-regression/)
  * introduction - [https://towardsdatascience.com/linear-regression-the-actually-complete-introduction-67152323fcf2](https://towardsdatascience.com/linear-regression-the-actually-complete-introduction-67152323fcf2)
  * code - [https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)

 See also [L], [Classification], [Multiple Linear Regression], [Non-Linear Regression], [Prediction Error], [Regression]


# Linear Temporal Logic

 Temporal logic is a subfield of mathematical logic that deals with reasoning about time and the temporal relationships between events. In artificial intelligence, temporal logic is used as a formal language to describe and reason about the temporal behavior of systems and processes.

 More at:
  * [https://www.geeksforgeeks.org/aritificial-intelligence-temporal-logic/](https://www.geeksforgeeks.org/aritificial-intelligence-temporal-logic/)
  * wikipedia - [https://en.wikipedia.org/wiki/Linear_temporal_logic](https://en.wikipedia.org/wiki/Linear_temporal_logic)

 See also [L], ...


# Link Prediction

 There are many ways to solve problems in [recommendation engines]. These solutions range from algorithmic approaches, link prediction algorithms, embedding based solutions, etc. Link prediction is also referred to as graph completion, a common problem in graph theory. In the simplest form, given a network, you want to know if there should be an edge between a pair of nodes. This definition changes slightly depending on the type of network you’re working with. A directed / multi graph can have slightly different interpretations but the fundamental concept of identifying missing edges in a network remains.

 ![]( {{site.assets}}/l/link_prediction.webp ){: width="100%"}

 Problems in link prediction are also quite common when dealing with temporal networks (networks which change over time). Given a network G at time step t, you would want to predict the edges of the graph G at time step t+1.

 {% youtube "https://www.youtube.com/watch?v=kq_b0QmxFCI" %}

 More at:
  * [https://towardsdatascience.com/link-prediction-recommendation-engines-with-node2vec-c97c429351a8](https://towardsdatascience.com/link-prediction-recommendation-engines-with-node2vec-c97c429351a8)

 See also [L], ...


# Linux Foundation AI And Data
# LFAI&Data

 The mission of LF AI & Data is to build and support an open artificial intelligence (AI) and data community, and drive open source innovation in the AI and data domains by enabling collaboration and the creation of new opportunities for all the members of the community.

 Projects
  * Graduated
   * [Milvus Database]
   * [ONNX Format]
   * Egeria, Flyte, Horovod, Pyro
  * Incubation
  * Sandbox


 More at:
  * site - [https://lfaidata.foundation/](https://lfaidata.foundation/) 
  * projects - [https://lfaidata.foundation/projects/](https://lfaidata.foundation/projects/)

 See also [L], ...


# Liquid Neural Network
# LNN

 {% youtube "https://youtu.be/0FNkrjVIcuk?si=I35p6esxM83rYsLf" %}

 {% youtube "https://www.youtube.com/watch?v=ql3ETcRDMEM" %}

 More at:
  * ...

 See also [L], ...


# LLaMA Model

 Using the scaling method described in [Chinchilla][Chinchilla Model]
 65 Billion parameters.

 {% youtube "https://www.youtube.com/watch?v=E5OnoYF2oAk" %}

 {% pdf "{{site.assets}}/l/llama_model_paper.pdf" %}

 More at:
  * LLaMa 2 
    * UI - [https://labs.perplexity.ai/](https://labs.perplexity.ai/)
  * LLaMa 1
    * announcement - [https://ai.facebook.com/blog/large-language-model-llama-meta-ai/](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    * paper [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
    * model card - [https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
    * model leak - [https://www.vice.com/en/article/xgwqgw/facebooks-powerful-large-language-model-leaks-online-4chan-llama](https://www.vice.com/en/article/xgwqgw/facebooks-powerful-large-language-model-leaks-online-4chan-llama)

 See also [L], ...


# LLaMA-Adapter Model

 We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune [LLaMA][LLaMA Model] into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the input text tokens at higher transformer layers. Then, a zero-init attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge. With efficient training, LLaMA-Adapter generates high-quality responses, comparable to Alpaca with fully fine-tuned 7B parameters. Furthermore, our approach can be simply extended to multi-modal input, e.g., images, for image-conditioned [LLaMA][LLaMa Model], which achieves superior reasoning capacity on [ScienceQA][ScienceQA Dataset].

 {% pdf "https://arxiv.org/pdf/2303.16199.pdf" %} 

 More at:
  * paper - [https://arxiv.org/abs/2303.16199](https://arxiv.org/abs/2303.16199)

 See also [L], ...


# Local Sensitive Hashing
# LSH

 ~ an algorithm used in [similarity search]

 a set of methods that is used to reduce the search scope by transforming data vectors into hash values while preserving information about their similarity.

 More at:
  * [https://towardsdatascience.com/similarity-search-part-5-locality-sensitive-hashing-lsh-76ae4b388203](https://towardsdatascience.com/similarity-search-part-5-locality-sensitive-hashing-lsh-76ae4b388203)
  * [https://srivatssan.medium.com/locality-sensitive-hashing-e70985c4e95d](https://srivatssan.medium.com/locality-sensitive-hashing-e70985c4e95d)

 See also [L], ...


# Log Loss Function

 See [Binary Cross-Entropy Loss Function]


# Log Transformation

 A [Feature Distribution Transformation]

 {% youtube "https://www.youtube.com/watch?v=LCDiQxB5S84" %}

 See also [L], ...

# Logical Reasoning

 If-then-else rules used in [expert systems][Expert System]

 ```
# Knowledge base
All men are mortal
# Input
Aristotle is a men
# Inference
==>
# New fact
Aristotle is mortal!

# If Aristotle is man AND all men are mortal, then Aritotle is mortal!
 ```

 Ex: personal assistant with memory and can infer from dialog new things (i.e graph network?) !

 See also [L], [Reasoning]


# Logistic Regression

 Not how long it will take for my car to stop given my speed (linear regression), but whether I am going to hit the tree or not (logistic regression). used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one.

 ![]( {{site.assets}}/l/logistic_regression_data.png ){: width="100%"}

 ![]( {{site.assets}}/l/logistic_regression_fitting.png ){: width="100%"}

 Beware:
  * To turn a probability into a classification, we need to use a threshold (P>0.5 or P<0.5)!
   * What about using a different P threshold? ==> multiple confusion matrix ==> ROC Curve

 See also [L], [ML Algorithm Evaluation], [Regression], [ROC Curve]


# Long Short-Term Memory Cell

# LSTM Cell

 * overview
  * input signal = previous state + new info
  * blue activation function = sigmoid activation function = switch (keep or forget, impact or no-impact)
  * red activation function = tanh --> add, no effect, or substract
 * cell state = highway that transfers information down to the sequence chain = memory of the network
 * gates
  * forget gate = decide which information should be thrown (=0) out or kept (=1) away (information = previous state + new input info) (sigmoid = 1 --> keep or = 0 forget!)
  * input gate = update the cell state with (transformed) input signal
  * output gate used to compute the hidden state = tanh(cell state) gated by input signal

 ![]( {{site.assets}}/l/long_short_term_memory_cell.png ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=YCzL96nL7j0" %}

 {% youtube "https://www.youtube.com/watch?v=8HyCNIVRbSU" %}

 {% youtube "https://www.youtube.com/watch?v=S27pHKBEp30" %}

 See also [L], [Hidden State], [LSTM Network]


# Long Short-Term Memory Network

# LSTM Network

 `A multi-layer Recurrent Neural Network, aka RNN, where a neuron is feeding its output to self, remembers its previous output. Good for sequences`. Used in speech recognition, Text to speech, handwriting recognition. Started becoming widespread in 2007. They are a type of Recurrent Neural Network that can efficiently learn via gradient descent. Using a gating mechanism, LSTMs are able to recognise and encode (short and very) long-term patterns (basic RNN can only remember a given length, i.e have short term memory because of vanishing gradient problem). LSTMs are extremely useful to solve problems where the network has to remember information for a long period of time as is the case in music and text generation.
 
 LSTMs also have the RNN chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

 ![]( {{site.assets}}/l/long_short_term_memory_unrolled.png ){: width="100%"}

 with

 ![]( {{site.assets}}/l/long_short_term_memory_repeating_module.png ){: width="100%"}

 In its chain, a LSTM can optionally use a Gated Recurrent Unit (GRU) cell, which is simpler than the one represented above.

 ```
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
 ```

 {% youtube "https://www.youtube.com/watch?v=WCUNPb-5EYI" %}

 {% pdf "{{site.assets}}/l/long_short_term_memory_paper.pdf" %}

 Beware:
  * `Are now deprecated by attention-based models, such as transformers? Yes`

 More at
  * [https://en.wikipedia.org/wiki/Long_short-term_memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
  * LSTM code
    * pytorch - [https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial](https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial)
    * keras - [https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
  * [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

 See also [L], [Attention-Based Model], [Gated Recurrent Unit Cell], [Gradient Descent Algorithm], [Recurrent Neural Network], [Transformer Model], [Vanishing Gradient Problem]


# Loss Function

 Loss function is a way to encode a goal. That loss function is going to dictate the optimized path toward that goal? Optimization?

 In most cases, the loss function is used for parameter estimation. Those parameters reflect the goal?

 `The loss function must encode what you want your model to do!` The loss function will take two items as input: the output value of our model and the ground truth expected value. The output of the loss function is called the loss which is a measure of how well our model did at predicting the outcome. A high value for the loss means our model performed very poorly. A low value for the loss means our model performed very well. In most learning networks, error is calculated as the difference between the actual output y and the predicted output ŷ. The function that is used to compute this error is known as Loss Function also known as Cost function. The loss function allows us to find the best line. The model is iterated to minimize the loss function using the gradient descent algorithm. Selection of the proper loss function is critical for training an accurate model. Certain loss functions will have certain properties and help your model learn in a specific way. Some may put more weight on outliers, others on the majority.

 The most common loss functions are:
  * [Mean Squared Error (MSE)][MSE] - Used in a linear regression, the best line is the one that minimize the root-mean square of the error.
  * [Mean Absolute Error (MAE)][MAE] - Use the absolute error instead of the RMS error. Beware of [outliers].
  * [Hinge Loss Function]
  * [Huber Loss Function] - Use the [MSE] for small values and [MAE] for large values ?
  * [0-1 Loss Function] : 0=correct 1=not-correct classification
  * [Binary cross-entropy loss function] (aka Log loss function) : Used with logistic regression because the logistic regression function (sigmoid or ?) is not linear and loss function needs to have a single minimum
  * [Cross-entropy loss function]
  * [Contrastive loss function] and [triplet loss function]
  * another custom function !
 Choose your loss function based on
  * the original estimator function (?) e.g. linear or sigmoid
  * must have a global minimum and not local ones

 More at :
  * choosing a loss function - [https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

 See also [L], [Activation Function], [Backpropagation], [Discriminator], [Gradient Descent Algorithm], [Linear Regression], [Optimizer], [Prediction Error], [Representation Space], [Residual]


# Loss Graph

 ![]( {{site.assets}}/l/loss_graph.png ){: width="100%"}

 See also [L], [Discriminator  Loss], [Generator Loss], [Loss Function]


# Low-Rank Adaptation Fine-Tuning

# LoRA Fine-Tuning

 A method for [parameter-efficient fine-tuning (PEFT)][PEFT]

 LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike [adapters], no additional inference latency

 {% youtube "https://www.youtube.com/watch?v=dA-NhCtrrVE" %}

 {% youtube "https://www.youtube.com/watch?v=iYr1xZn26R8" %}

 {% pdf "https://arxiv.org/pdf/2106.09685.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
  * article(s)
    * [https://bdtechtalks.com/2023/05/22/what-is-lora/](https://bdtechtalks.com/2023/05/22/what-is-lora/)

 See also [L], [QLoRA Fine-Tuning]


# Low-Rank Approximation

 Replace a high-rank matrix by an approximation returned by the multiplication of 2 low-rank matrices.
 To find the best low-rank approximation use [Single Value Decomposition (SVD)][SVD]!

 ```
 Bm,n = Am,k  .  Ck,n

# k << n  and k << m
# Am,k = matrix of m rows and k columns
# Ck,n = matrix of k rows and n columns
 ```

 To find the optimum values for k, Am,k , and Ck,n look at [singular value decomposition]

 {% youtube "https://www.youtube.com/watch?v=12K5aydB9cQ" %}

 See also [L], ...
