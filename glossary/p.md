---
title: P
permalink: /p/

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


# Pandas Python Module

 A [Python module] for importing, transforming, and working with tabular data

 More at:
  * home - [https://pandas.pydata.org/](https://pandas.pydata.org/)
  * user guide - [https://pandas.pydata.org/docs/user_guide/index.html](https://pandas.pydata.org/docs/user_guide/index.html)
  * API reference - [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

 See also [P], ...


# Parameter

 A model parameter is something that the ML can learn from the data. For example, the weight of an input in a perceptron. Indeed the model has to have parameters to make predictions. This "parameters" are not set by humans. Hyperparameters cannot be learned from the data and are set by humans. Ex: number of layers in the neural network. 
  * GPT-3 possesses 175 billion weights connecting the equivalent of 8.3 million neurons arranged 384 layers deep.

 See also [P], [Hyperparameter], [Parametric Knowledge]


# Parameter-Efficient and Quantization-Aware Adaption
# PEQA

 [Parameter-efficient fine-tuning (PEFT)][PEFT] methods have emerged to mitigate the prohibitive cost of full fine-tuning large language models (LLMs). Nonetheless, the enormous size of LLMs impedes routine deployment. To address the issue, we present Parameter-Efficient and Quantization-aware Adaptation (PEQA), a novel quantization-aware [PEFT] technique that facilitates model compression and accelerates inference. PEQA operates through a dual-stage process: initially, the parameter matrix of each fully-connected layer undergoes quantization into a matrix of low-bit integers and a scalar vector; subsequently, fine-tuning occurs on the scalar vector for each downstream task. Such a strategy compresses the size of the model considerably, leading to a lower inference latency upon deployment and a reduction in the overall memory required. At the same time, fast fine-tuning and efficient task switching becomes possible. In this way, PEQA offers the benefits of quantization, while inheriting the advantages of [PEFT]. We compare PEQA with competitive baselines in comprehensive experiments ranging from natural language understanding to generation benchmarks. This is done using large language models of up to 65 billion parameters, demonstrating PEQA's scalability, task-specific adaptation performance, and ability to follow instructions, even in extremely low-bit settings.

 {% pdf "https://arxiv.org/pdf/2305.14152.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2305.14152(https://arxiv.org/abs/2305.14152)
  * articles
    * [https://bdtechtalks.com/2023/09/18/what-is-llm-compression/](https://bdtechtalks.com/2023/09/18/what-is-llm-compression/)

 See also [P], [QLoRA]


# Parameter-Efficient Fine-Tuning
# PEFT

 Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. Recent State-of-the-Art PEFT techniques achieve performance comparable to that of full fine-tuning.

 Methods
  * [Low-Rank Adaptation (LoRA)][LoRA] of [LLMs]
  * [Prefix Tuning]
  * [P-Tuning]
  * [Prompt Tuning]
  * [AdaLoRA Tuning]
  * with [model compression]
    * [Parameter-Efficiient and Quantization-Aware Adaptation][PEQA]
    * [QLoRA]

 More at:
  * [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

 See also [P], ...


# Parametric Knowledge

 ~ knowledge stored in the parameters of the model. Frozen in time.

 See also [P], [Parameter]


# Particule Swarm Optimization Algorithm
# PSO Algorithm

 PSO was first intended for simulating social behaviour, as a stylized representation of the movement of organisms in a bird flock or fish school. The algorithm was simplified and it was observed to be performing optimization. 

 PSO is a [metaheuristic] as it makes few or no assumptions about the problem being optimized and can search very large spaces of candidate solutions. Also, PSO does not use the gradient of the problem being optimized, which means PSO does not require that the optimization problem be differentiable as is required by classic optimization methods such as [gradient descent] and quasi-newton methods. However, [metaheuristics] such as PSO do not guarantee an optimal solution is ever found.

 ![]( {{site.assets}}/p/particle_swarm_optimization.gif ){: width="100%"}

 {% youtube "https://www.youtube.com/watch?v=8xycqWWqz50" %}
 {% youtube "https://www.youtube.com/watch?v=JhgDMAm-imI" %}

 More at:
  * Articles
    * [https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/](https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/)
    * wikipedia - [https://en.wikipedia.org/wiki/Particle_swarm_optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
    * code
      * race line - [https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO](https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO)

 See also [P], ...


# Passive Learning

 The main hypothesis in active learning is that if a learning algorithm can choose the data it wants to learn from, it can perform better than traditional methods with substantially less data for training. But what are these traditional methods exactly? These are tasks which involve gathering a large amount of data randomly sampled from the underlying distribution and using this large dataset to train a model that can perform some sort of prediction. You will call this typical method passive learning. One of the more time-consuming tasks in passive learning is collecting labelled data. In many settings, there can be limiting factors that hamper gathering large amounts of labelled data.

 See also [P], [Active Learning], [Random Sampling]


# Pathways Language Model
# PaLM Model

 {% pdf "{{site.assets}}/p/palm_model_paper.pdf" %}

 More at :
  * [https://medium.com/@tech_optimist/palm-on-my-forehead-not-another-large-language-model-6dddd641211b](https://medium.com/@tech_optimist/palm-on-my-forehead-not-another-large-language-model-6dddd641211b)

 See also [P], [Chain Of Thought Prompting], [Pathways Model Architecture]


# Pathways Language Model Embodied Model
# PaLM-E Model

 An embodied multimodal language model developed by [Google] and based on the existing [PaLM Model]

 {% youtube "https://www.youtube.com/watch?v=2BYC4_MMs8I" %}
 
 More at:
  * [https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html](https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html)

 See also [P], [Pathways Model Architecture]


# Pathways Model Architecture

 An architecture developed by [Google] to support (1) transfer learning, (2) multimodal learning, (3) Sparse activation, i.e NOT dense networks/models

  {% youtube "https://www.youtube.com/watch?v=Nf-d9CcEZ2w" %}

  More at 
   * [https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/)

  See also [P], [Multimodal Learning], [PaLM Model], [Sparse Activation]


# Pattern Recognition

 Pattern recognition is the automated recognition of patterns and regularities in data. It has applications in statistical data analysis, signal processing, image analysis, information retrieval, bioinformatics, data compression, computer graphics and machine learning. Pattern recognition has its origins in statistics and engineering; some modern approaches to pattern recognition include the use of machine learning, due to the increased availability of big data and a new abundance of processing power.

 More at:
  * quickdraw - [https://experiments.withgoogle.com/quick-draw](https://experiments.withgoogle.com/quick-draw)
  * [https://en.wikipedia.org/wiki/Pattern_recognition](https://en.wikipedia.org/wiki/Pattern_recognition)

 See also [P], [Hand Gesture Recognition]


# People

 People
  * [Alan Turing] - A founder of AI
  * [Alex Krizhevsky] - Build [AlexNet] and creator of [CIFAR Datasets]
  * [Andrew Ng] - Cofounder and head of [Google] Brain and was the former Chief Scientist at [Baidu]
  * [Bill Gates] - Founder and now chairman at [Microsoft]
  * [David Luan] - CEO of [Adept]
  * [Elon Musk] - CEO of Tesla
  * [Eric Schmidt] - Chairman of Alphabet / [Google]
  * [Fei-Fei Li] - Creator of the [ImageNet dataset], focus on the data, not the algorithm!
  * [Geoffrey Hinton] - Lead his student with [AlexNet], a godfather of AI and [Deep Learning]. Turing award in 2018.
  * [Greg Brockman] - Co-founder of [OpenAI]
  * [Ilya Sutskever] - Co-founder of [OpenAI]
  * [Sam Altman] - CEO of [OpenAI]
  * [Sundar Pichai] - CEO of Alphabet/[Google]
  * [Yann LeCun] - Turing award in 2018 for work on [Deep Learning]
  * [Yoshua Bengio] - Professor at the Department of Computer Science at the Université de Montréal. Turing award in 2018 for work on [Deep Learning]
  * ...

  Others
   * Manuela Veloso - Carnegie Mellon University and Head of research at JPMC

 See also [P], [AI Movie], [Company]


# Pepper Robot

 Robot built by [Softbank Robotics]

 {% youtube "https://www.youtube.com/watch?v=Ti4NiaQj8q0" %}

 More at:
  * [https://us.softbankrobotics.com/pepper](https://us.softbankrobotics.com/pepper)

 See also [P], ...


# Perceiver IO Model

 Product arbitrary size outputs - reconstructing the input

 ![]( {{site.assets}}/p/perceiver_io_model.png ){: width="100%"}


 See also [P], [Attention-Based Model], [Perceiver Model]


# Perceiver Model

  * convert input to simple 2D byte array
  * Encode information about the input array using a smaller number of latent feature vectors using transformer-style attention
  * final aggregation down to a category label
  . :warning: Used for classification

 ![]( {{site.assets}}/p/perceiver_model.png ){: width="100%"}

 See also [P], [Attention-Based model], [Transformer Model]


# Perceptron

 A neural network consisting of only 1 layer and 1 neuron.

 Note that a perceptron is a prototype of a modern [artificial neuron] , except it does not have an [activation function] ? Not sure!

 The perceptron was invented in 1943 by McCulloch and Pitts.

 In machine learning, the perceptron (or McCulloch-Pitts neuron) is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.[1] It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

 More at:
  * [https://en.wikipedia.org/wiki/Perceptron](https://en.wikipedia.org/wiki/Perceptron)
  * inventor - [https://en.wikipedia.org/wiki/Frank_Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt)
  * book ? - [https://direct.mit.edu/books/book/3132/PerceptronsAn-Introduction-to-Computational](https://direct.mit.edu/books/book/3132/PerceptronsAn-Introduction-to-Computational)

 See also [P], ...


# Perfect Information

 In [Reinforcement Learning (RL)][RL], an environment where everything is known.

 Example:
  * In chess, by looking at the board, we can see the position of all pieces and therefore find the optimal decision

 In an imperfect information game, we have to make assumption and associate probabilities to those assumption.

 More at:
  * ...

 See also [P], ...


# Perplexity

 Used with language model. The smaller its value, the better.

 {% youtube "https://www.youtube.com/watch?v=NURcDHhYe98" %}

 See also [P], [Cross-Entropy], [Entropy], [Generative Model], 


# Personal Assistant

 * Pi by [Inflection AI]

 See also [P], [Logic Model]


# Phenaki Model

 A model for generating videos from text, with prompts that can change over time, and videos that can be as long as multiple minutes.
 Built by employees at [Google]

 {% pdf "{{site.assets}}/p/phenaki_model_paper.pdf" %}

 More at:
  * home - [https://phenaki.video/](https://phenaki.video/)
  * paper - [https://openreview.net/forum?id=vOEXS39nOF](https://openreview.net/forum?id=vOEXS39nOF)


# Photonic Chip

 {%  youtube "https://www.youtube.com/watch?v=IwzguEPIddU" %}

 More at:
  * Companies
    * [https://lightmatter.co/](https://lightmatter.co/)
    * [https://saliencelabs.ai/](https://saliencelabs.ai/)
  * papers
    * [https://www.science.org/doi/10.1126/science.ade8450](https://www.science.org/doi/10.1126/science.ade8450)


 See also [P], ...


# Piano Roll

 As a 2D matrix, also known as a piano roll, with time on the horizontal and pitch on the vertical axis.

 See also [P], [U-Net Architecture]


# Picasso Model

 Build by [Nvidia]

 More at:
  * [https://www.creativebloq.com/news/nvidia-picasso-ai](https://www.creativebloq.com/news/nvidia-picasso-ai)

 See also [N], ...


# Pinecone Company

 A company that is building a commercial [vector database], similar to [Milvus][Milvus Database]

 More at:
  * home - [https://www.pinecone.io/](https://www.pinecone.io/)
  * articles
   * [https://www.pinecone.io/learn/series-b/](https://www.pinecone.io/learn/series-b/)

 See also [P], ...


# Pinecone Database

 More at:
  * docs - [https://docs.pinecone.io/docs/overview](https://docs.pinecone.io/docs/overview)

 See also [P], ...


# Pipe Mode

 The training set is streamed all the way to the inference point.

 See also [P], [File Mode]


# Pixel RNN

 [Autoregressive models][Autoregressive Model] such as PixelRNN instead train a network that models the conditional distribution of every individual pixel given previous pixels (to the left and to the top). This is similar to plugging the pixels of the image into a char-rnn, but the RNNs run both horizontally and vertically over the image instead of just a 1D sequence of characters. PixelRNNs have a very simple and stable training process (softmax loss) and currently give the best log likelihoods (that is, plausibility of the generated data). However, they are relatively inefficient during sampling and don’t easily provide simple low-dimensional codes for images.

 More at:
  * paper - [https://arxiv.org/abs/1601.06759](https://arxiv.org/abs/1601.06759)

 See also [P], [RNN]


# Pixel Space

 In the pixel space, operations are done based on the values/parameters of pixels.

 See also [P], [Latent Space], [Space], 


# Plagiarism Checker

 More at:
  * [https://quillbot.com/plagiarism-checker](https://quillbot.com/plagiarism-checker)
  * [https://www.grammarly.com/plagiarism-checker](https://www.grammarly.com/plagiarism-checker)

 See also [P], [ChatGPT Model]


# Pluribus Model

 Pluribus is a computer poker player using artificial intelligence built by [Meta]'s AI Lab and [Carnegie Mellon University]. Pluribus plays the poker variation no-limit Texas hold 'em and is "the first bot to beat humans in a complex multiplayer competition".

 Challenge:
  * Poken is a game of imperfect information

 {% pdf "{{site.assets}}/p/pluribus_science_article.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=u90TbxK7VEA" %}

 {% youtube "https://www.youtube.com/watch?v=BDF528wSKl8" %}

 More at:
  * [https://ai.facebook.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/](https://ai.facebook.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/)
  * [https://en.wikipedia.org/wiki/Pluribus_(poker_bot)](https://en.wikipedia.org/wiki/Pluribus_(poker_bot))
  * [https://www.smithsonianmag.com/smart-news/poker-playing-ai-knows-when-hold-em-when-fold-em-180972643/](https://www.smithsonianmag.com/smart-news/poker-playing-ai-knows-when-hold-em-when-fold-em-180972643/)

 See also [P], [Game Theory]


# Point Estimator

 This definition of a point estimator is very general and allows the designer of an estimator great flexibility. While almost any function thus qualifies as an estimator, a good estimator is a function whose output is close to the true underlying θ that generated the training data. Point estimation can also refer to estimation of relationship between input and target variables referred to as function estimation.

 See also [P], [Estimator], [Function Estimation]


# Point-E Model

 Text-to-3d using 2D diffusion ?

 While recent work on text-conditional 3D object generation has shown promising results, the state-of-the-art methods typically require multiple GPU-hours to produce a single sample. This is in stark contrast to state-of-the-art generative image models, which produce samples in a number of seconds or minutes. In this paper, we explore an alternative method for 3D object generation which produces 3D models in only 1-2 minutes on a single GPU. Our method first generates a single synthetic view using a text-to-image diffusion model, and then produces a 3D point cloud using a second diffusion model which conditions on the generated image. While our method still falls short of the state-of-the-art in terms of sample quality, it is one to two orders of magnitude faster to sample from, offering a practical trade-off for some use cases. We release our pre-trained point cloud diffusion models, as well as evaluation code and models, at this https URL.

 {% pdf "https://arxiv.org/pdf/2212.08751.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2212.08751](https://arxiv.org/abs/2212.08751)
  * code - [https://github.com/openai/point-e](https://github.com/openai/point-e)
  * blog - [https://the-decoder.com/point-e-openai-shows-dall-e-for-3d-models/](https://the-decoder.com/point-e-openai-shows-dall-e-for-3d-models/)
  * articles
   * [https://techcrunch.com/2022/12/20/openai-releases-point-e-an-ai-that-generates-3d-models/](https://techcrunch.com/2022/12/20/openai-releases-point-e-an-ai-that-generates-3d-models/)

 See also [P], [CLIP Model], [DALL-E Model], [DreamFusion Model]


# Policy

 Policy refers to the strategy the agent follows to determine the next action based on current state. The policy may look like a lookup table, a simple function, or it may involve extensive computation such as a search process. Also, the policy alone is sufficient to determine the agent’s behavior.


 State to action function !
 * Strategy of agent in pursuit of goal
 * Policy is optimal if its expected reward >= any other policy for all state

 Policy types
  * Take first action in mind
  * Select action at random
  * use heuristic

 Policy characteristic
  * agent's policy change due to ...
  * stochastic ==> proba for each action
  * deterministic ==> return the chosen action

 Greedy policy = agent exploits the current knowledge 

 See also [P], [Optimal Policy]


# Policy Evaluation

 It computes values for the states in the environment using the policy provided by the policy improvement phase.

 See also [P], ...


# Policy Function

 In [Reinforcement Learning], a policy is a function that takes for input a state and outputs an action.

 ```
a deterministic policy π is a function that takes as an input a state “S” and returns an action “a” 
That is: π(s) → a
 ```

 Policies are usually stochastic, meaning that we select an action from a probability distribution. As an example, imagine that we are state S2 again. The policy will not just tell the agent “take action a6”. Instead, it will say “take action a6 with probability 88%, and take action a3 with probability 12%”.

 a stochastic policy π is a function that takes as an input a state "S" and returns a set of action A with, for each action, an associated probability.
 
 In the case of Deep RL, a policy function is an artificial neural network.

 During training, updated at every iteration. The goal of the reinforcement learning in AWS DeepRacer is to learn the optimal policy in a given environment. Learning is an iterative process of trials and errors. Note:
  * Firstly, the reinforcement learning agents starts with a random policy π (i). Policy Evaluation will evaluate the value functions like state values for that particular policy.
  * The policy improvement will improve the policy and give us π (1) and so on until we get the optimal policy where the algorithm stops. This algorithm communicates back and forth between the two phases—Policy Improvement gives the policy to the policy evaluation module which computes values.

 In reinforcement learning, a policy function defines an agent's behavior by mapping states to actions. It specifies which action the agent should take in any given state. Some key characteristics:
  * The policy is the core element that guides how an [RL agent] acts. It fully defines the agent's behavior.
  * It maps states to probability distributions over possible actions.
  * Stochastic policies return probabilities for each action. Deterministic policies return the chosen action.
  * Policies aim to maximize reward signals from the environment over time.
  * Policies can be learned through methods like policy gradients, which optimize network parameters.
  * Learned policies start out random but improve through experience and updating rewards.
  * Simple policies may implement hand-coded rules. Complex ones use neural networks.
  * The input is a state description. The output is which action to take.
  * Better policies lead to higher accumulated rewards over time. The optimal policy maximizes total reward.
  * Policies balance exploration of new states with exploitation of known rewards.
  * The policy function represents the agent's complete behavior model. Changing the policy changes how the agent acts.

 So in summary, a policy function in reinforcement learning maps state information to agent actions in order to define behavior that maximizes rewards from the environment over time. The policy is the core component that is learned.

 See also [P], ...


# Policy Gradient Algorithm

 A set of algorithms that update the policy [artificial neural network].

 Examples of algorithm:
  * [REINFORCE algorithm]
  * [Advanced Actor-Critic (A2C)][A2C]
  * [Asynchronous Advanced Actor-Critic (A3C)][A3C]
  * [Deep Deterministic Policy Gradient (DDPG)][DDPG]
  * [Proximal Policy Optimization (PPO)][PPO]


 {% youtube "https://www.youtube.com/watch?v=YOW8m2YGtRg" %}

 {% youtube "https://www.youtube.com/watch?v=tqrcjHuNdmQ" %}

 {% youtube "https://www.youtube.com/watch?v=AKbX1Zvo7r8" %}

 More at : 
  * [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)
  * [https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
  * colab - [https://colab.research.google.com/github/DJCordhose/ai/blob/master/notebooks/rl/pg-from-scratch.ipynb](https://colab.research.google.com/github/DJCordhose/ai/blob/master/notebooks/rl/pg-from-scratch.ipynb)

 See also [P], [Proximal Policy Optimization Algorithm], [Reinforcement Learning]


# Policy Neural Network

 In [Reinforcement Learning], in the case of [AWS DeepRacer], the policy network consists of a [CNN] followed by a neural network to turn the features into an [Action] taken from the [Action Space]. In this case, the policy network can be though off as a classifier, to turn an image into an action.

 ![]( {{site.assets}}/p/policy_neural_network.png ){: width="100%}

 See also [P], ...


# Polynomial Regression

 A form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. Examples:
   * cubic regression
   * quadratic regression

 ![]( {{site.assets}}/p/polynomial_regression.png ){: width="100%"}

 More at:
  * polynomial regression - [https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de](https://towardsdatascience.com/polynomial-regression-the-only-introduction-youll-need-49a6fb2b86de)
  * code - [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#examples-using-sklearn-preprocessing-polynomialfeatures)

 See also [P], [Cubic Regression], [Quadratic Regression], [Regression]


# Pose Estimation

 Can be estimated by [YOLO models][YOLO Model]

 More at:
  * [https://docs.ultralytics.com/tasks/pose/](https://docs.ultralytics.com/tasks/pose/)
  * Teachable Machine project - [https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491](https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491)

 See also [P], ...


# Positional Encoding

 When a model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the words/tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed. Example of positional encoding formula:

 ![]( {{site.assets}}/p/positional_encoding_formula.png ){: width="100%"}

 See also [P], [Multi-Head Attention]


# Positive And Negative Pairs

 ~ used in embedded space. Positive pairs = points should be brought closer, negative pair = points should be brought further apart

 Positive and negative pairs are concepts often used in machine learning, particularly in tasks related to similarity or ranking. These pairs are used to represent relationships between data points and help models learn to distinguish between different classes or levels of similarity.

  * Positive Pairs: Positive pairs, also known as matching pairs, are pairs of data points that are considered similar or belong to the same class. In other words, these pairs represent instances that should be recognized as related by a model. For example, in face recognition, positive pairs might consist of images of the same person taken from different angles or under different lighting conditions.
  * Negative Pairs: Negative pairs, also known as non-matching pairs or contrasting pairs, are pairs of data points that are considered dissimilar or belong to different classes. These pairs represent instances that should be distinguished from each other by a model. Continuing with the face recognition example, negative pairs might consist of images of different individuals.

 Positive and negative pairs are commonly used in various machine learning tasks, such as:
  * Siamese Networks: These are neural network architectures designed to learn similarity or dissimilarity between data points. Siamese networks use positive and negative pairs during training to learn to minimize the distance between positive pairs and maximize the distance between negative pairs in the learned representation space.
  * Triplet Loss: Triplet loss is another approach used in tasks like face recognition. It involves using three data points: an anchor A, a positive P (similar to the anchor), and a negative N (dissimilar to the anchor). The network is trained to minimize the distance between the anchor and positive while maximizing the distance between the anchor and negative.
  * Ranking and Retrieval: In information retrieval tasks, positive pairs could be queries matched with relevant documents, while negative pairs could be queries matched with non-relevant documents. Models learn to rank relevant documents higher than non-relevant ones.

 The use of positive and negative pairs helps models learn meaningful representations or decision boundaries that can generalize well to new, unseen data. By incorporating both similarity and dissimilarity information, models can learn to make accurate predictions or rankings in various tasks.

 An alternative technique is [ranking]

 More at:
  * ...

 See also [P], ...


# Positive Attribute

 A cat has legs, fur, 2 eyes, etc. Those are positive attributes. 

 See also [P], [Attribute], [Negative Attribute]


# Post-Training Quantization
# PTQ

 A [quantization] method that involves transforming the parameters of the LLM to lower-precision data types after the model is trained. PTQ aims to reduce the model’s complexity without altering the architecture or retraining the model. Its main advantage is its simplicity and efficiency because it does not require any additional training. But it may not preserve the original model’s accuracy as effectively as the other techniques.

 More at:
  * [https://bdtechtalks.com/2023/09/18/what-is-llm-compression/](https://bdtechtalks.com/2023/09/18/what-is-llm-compression/)

 See also [P], [Model Compression]


# Posterior Belief

 Posterior belief refers to the degree of belief in a hypothesis or claim after accounting for observed evidence. It is the result of updating your [prior beliefs] using [Bayes' theorem] to incorporate the likelihood of the evidence.

 More at:
  * ...

 See also [P], ...


# Precision

 Metric used for [model evaluation] when the cost of false positives is high. An example task could be spam detection, when we don't want to incorrectly classify legitimate emails as spam.

 Precision is the fraction of the tumors that were predicted to be malignant (of one call) that are actually malignant (of that class).
 
 ```
# TP : a cat is recognized as a cat
# FP : a cat is recognized as a dog

               TP               # samples in class that are correctly identified       
Precision = --------- =    --------------------------------------------------------
             TP + FP                    # sample in class
 ```

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [P], [Confusion Matrix]


# Prediction

 The desired outcome of Machine Learning. Those predictions are based on hidden patterns in the training data. Low bias and low variance: every is close to the bulls eye. High bias and low variance: everything is clustered but with an offset from the bullseye (eg systematically to high). Low bias and high variance: appears to be centered on target but far from the bullseye. High variance and high bias: all over the place and not on target! Beware that the goal is not to minize the prediction error and not necessarily the bias and the variance.

 See also [P], [Bias], [Machine Learning], [Prediction Error], [Variance]


# Prediction Error

 To minimize the prediction error, you need a balancing act between the bias and the variance of the data.
 
 ![]( {{site.assets}}/p/prediction_error.png ){: width="100%"}
 
 ```
Prediction Error = actual_value - predicted_value
 ```

   * Relative Approximation Error
   * Root Mean Square
   * Mean Absolute Error
   * Mean Absolute Percentage Error

 ![]( {{site.assets}}/p/prediction_error_measurement.png ){: width="100%"}

 See also [P], [Bias], [Gradient Descent Algorithm], [Loss Function], [Mean Absolute Error], [Mean Absolute Percentage Error], [Prediction], [Relative Approximation Error], [Root Mean Square Error], [Variance]


# Predictive Maintenance

 Predict when a part will fail.

 See [P], [Regression], [Supervised Learning]


# Predictor Variable

 Input X / independent variable to estimate the dependent variable in regression.

 See also [P], [Regression], [Response Variable]


# Pretrained Model

 ~ a base model that support transfer learning. A pre-trained model, keeping with Gladwell’s 10,000 hours theory, is the first skill you develop that can help you acquire another one faster. For example, mastering the skill of solving math problems can help you more quickly acquire the skill of solving engineering problems. A pre-trained model is trained (by you or someone else) for a more general task and is then available to be fine-tuned for different tasks. Instead of building a model from scratch to solve your problem, you use the model trained on a more general problem as a starting point and give it more specific training in the area of your choice using a specially curated dataset. A pre-trained model may not be 100% accurate, but it saves you from reinventing the wheel, thus saving time and improving performance.

 Useful if 
  * retraining the model is expensive or time consuming
  * model architecture allow transfer learning (through fine-tuning). 
  . :warning: as far as I can tell, this is only possible today with transformer models (?)

 Examples of pretrained models are:
  * [BERT][BERT Model] whose [pretraining objectives] include [MLM] and [NSP] 
  * [RoBERTa][RoBERTa Model]
  * [GPT models][GPT Model]
  * ...

 See also [P], [Supervised Fine-Tuning], [Transfer Learning], [Transformer Model], [Upstream Task]


# Pretraining Objective

 The objectives on which the [pretrained model] was trained.

 See also [P], [RoBERTa Model]


# Principal Component Analysis
# PCA

 ~ an algorithm used for [dimensionality reduction]

 The most popular dimensionality reduction method is Principal Component Analysis (PCA), which reduces the dimension of the feature space by finding new vectors that maximize the linear variation of the data. Why use? many situations that require low dimensional data including:
  * data visualisation
  * data storage
  * computation

 PCA can reduce the dimension of the data dramatically and without losing too much information when the linear correlations of the data are strong (PCA is based onlinear algebra!). :warning: And in fact you can also measure the actual extent of the information loss and adjust accordingly.) Principal Component Analysis (PCA) is a very popular technique used by data scientists primarily for dimensionality reduction in numerous applications ranging from stock market prediction to medical image classification. Other uses of PCA include de-noising and feature extraction. PCA is also used as an exploratory data analysis tool. To better understand PCA let’s consider an example dataset composed of the properties of trucks. These properties describe each truck by its color, size, compactness, number of seats, number of doors, size of trunk, and so on. Many of these features measured will be redundant and therefore, we should remove these redundancies and describe each truck with fewer properties. This is precisely what PCA aims to do. PCA is a technique used to compress d features into `p<<d` features, while preserving as much of the information as possible (~ compression !). A classic demonstration for PCA is given with images. A black and white image can be represented as an n by d matrix of integers, determining the grayscale of each pixel. PCA provides a low-rank (low-dimension) representation of that matrix that can be stored with (n+d) p numbers, instead of nd, such that the compressed image looks almost the same as the original. In the context of machine learning (ML), PCA is a dimension reduction technique. When the number of features is large, ML algorithms either are at risk of overfitting, or require too much time to train. To that end, `PCA can reduce the input dimension. The way PCA reduces the dimension is based on correlations. Two features are correlated if, given the value of one, you can make an educated guess about the value of the other`. PCA, with a target dimension of p, finds p features such that a linear function of these will do the best job of predicting the original d features. This kind of information-preserving objective makes the output of PCA useful to downstream tasks.

 Pros:
  * Relatively computationally cheap.
  * Can save embedding model to then project new data points into the reduced space.

 Cons:
  * Linear reduction limits information that can be captured; not as discriminably clustered as other algorithms.

 More at:
  * [https://setosa.io/ev/principal-component-analysis/](https://setosa.io/ev/principal-component-analysis/)
  * [https://dimensionality-reduction-293e465c2a3443e8941b016d.vercel.app/](https://dimensionality-reduction-293e465c2a3443e8941b016d.vercel.app/)

 See also [P], [Feature Extraction], [Linear Autoencoder], [Linear Discriminant Analysis], [ML Algorithm], [Overfitting], [Synthesized Variable]


# Prior

 Can refer either to [prior knowledge] or [prior probability]

 See also [P], ...


# Prior Belief

 A prior belief refers to the initial degree of belief in a hypothesis or claim before accounting for observed data or evidence. It is the probability assigned to a hypothesis before new evidence is considered.

 Some key properties of prior beliefs:

  * Prior beliefs are subjective and based on previous knowledge, expertise, or assumptions made about a problem. They may also be called the prior probability.
  * In Bayesian statistics, the prior probability is the initial probability assigned to a hypothesis before new relevant data is collected.
  * A prior is needed to apply [Bayes' theorem] to update beliefs when new evidence is obtained.
  * Priors can come from previous data, logical constraints, subjective judgment, physical models, etc. Uninformative priors reflect limited initial knowledge.
  * The choice of prior can significantly impact conclusions if data is limited. More data makes inferences less dependent on the specific prior.
  * Priors are combined with the likelihood of observing evidence to determine the posterior probability using Bayes' rule.
  * As more evidence accumulates, the [posterior belief] dominates over the influence of the prior.
  * Common non-informative priors include the uniform distribution or Jeffreys prior which let data drive conclusions.

 In summary, prior beliefs represent initial hypotheses and uncertainties before considering new evidence. [Bayesian inference] provides a mathematical framework for updating priors to [posterior beliefs] as new data is collected.

 More at:
  * ...

 See also [P], ...


# Prior Knowledge

 `~ Prior knowledge about the world, enabling efficient decision making. Also the reason why human learn so much faster than computers!`. Fundamental concepts that are transfered from one task to the other that humans do not have to learn and therefore reduce the training time significantly.

 `~ stereotypes, shortcuts.`

 Examples of priors are
  * concepts of objects
  * look similar = act similar
  * object semantics = a key 
  * object properties = a monster can kill me, a key can open a door --> go to key before door
  * affordance : In HCI, to “afford” means to “invite” or to “suggest.” Ex: when you see a button, you assume you can push it to trigger an action. If the button is next to an elevator, you assume it is used to call the elevator. If you recognize a ladder, you assume you can climb it. etc
 . :warning: Note that many priors are innate (look at faces) while others are learned (object permanence = when an object disappear from my view, it still exists) 
 
 Beware priors can also slow down or even prevent learning
  * False Affordances. This type of affordance doesn’t have an apparent function when it is observed. That causes the individual to perceive possibilities of action that do not exist in reality. A placebo is an example of a false affordance.
  * Hidden Affordances. These affordances offer the potential for actions to be taken, but are not necessarily perceived by individuals within their environment. One might look at a book and think, “I could use that for reading.” It could also be used to smash a spider that looks like trouble.
  * Perceptible Affordances. These clues offer information to individuals that allows them to understand what actions can be taken and encourages them to take that action.A

 {% youtube "https://www.youtube.com/watch?v=Ol0-c9OE3VQ" %}
 
 {% pdf "https://arxiv.org/pdf/1802.10217.pdf" %}

 More at:
  * post - [https://rach0012.github.io/humanRL_website/](https://rach0012.github.io/humanRL_website/)
  * code - [https://github.com/rach0012/humanRL_prior_games](https://github.com/rach0012/humanRL_prior_games)
  * paper -  [https://arxiv.org/abs/1802.10217](https://arxiv.org/abs/1802.10217)

 See also [P], [Learning Rate], [Transfer Learning]


# Prior Probability

 Assess a (sample distribution) probability using the training data (i.e. the training data is representative of the data distributions or likelihoods)

 See also [P], [Prior Belief]


# Prioritized Experience Replay

 In [RL], Replay important transitions more frequently for efficient learning.

 Prioritized experience replay is a technique to improve the efficiency of experience replay in reinforcement learning. The key ideas are:

  * In standard experience replay, samples are drawn uniformly at random from the [replay memory].
  * In prioritized experience replay, transitions are sampled with [probability] proportional to a priority value.
  * Transitions that have high priority (e.g. errors in value function prediction) are sampled more frequently.
  * Priorities can be determined in different ways, but common approaches involve using the magnitude of error in the Q-value or TD error.
  * After sampling based on priority, importance sampling weights are used to correct for the non-uniform probabilities.
  * Periodically, priorities are updated based on the latest errors. So over time, the priorities shift to reflect new assessments.
  * This focuses sampling and replay on transitions deemed important either due to uncertainty or prediction error. This leads to faster and more efficient learning.
  * A potential downside is it could over-fit to certain transitions if priorities do not get updated properly. Proper tuning is required.

 So in summary, prioritized experience replay samples important transitions more frequently to make better use of experience in reinforcement learning. It is a key technique to enable efficient learning from experience.

 More at:
  * ...

 See also [P], ...

# Probabilistic Inference For Learning Control Model
# PILCO Model

 PILCO (Probabilistic Inference for Learning Control) is a [model-based reinforcement learning] method for continuous control problems:

  * It learns a probabilistic dynamics model of the environment based on Gaussian processes. This captures model uncertainties.
  * It uses this learned model to perform probabilistic inference and directly compute a controllers policy in closed form, avoiding needing to learn a value function.
  * Specifically, it uses the dynamics model to estimate the distributions of future states for different sequence of actions. It then optimizes these predictions to find an optimal policy that maximizes long-term rewards.
  * A key benefit is data efficiency - by learning a model, PILCO can plan solutions with fewer interactions with the real [environment]. It is therefore a [sample efficient RL Algorithm]
  * It also explicitly handles model uncertainty during long-term planning. This avoids overfitting to poor models.
  * Limitations are that it relies on Gaussian processes, which can be computationally expensive in high dimensions. The dynamics model learning also currently happens offline.
  * PILCO has been applied to control tasks like cartpole, pendulum, and robot arm control. It achieves good performance with relatively little training data.

 In summary, PILCO is a [model-based reinforcement learning] technique that learns a probabilistic dynamics model for efficient and robust [policy] optimization in continuous control problems. It explicitly represents and handles model uncertainties.

 More at:
  * ...

 See also [P], ...


# Probability

 Statistics and probability are two closely related fields, but they have distinct differences.

 Probability is the branch of mathematics that deals with the study of random events or phenomena. It provides a way to quantify the likelihood of an event occurring. Probability theory is used to make predictions about the likelihood of future events based on past observations and data. Probability is used in many areas such as finance, physics, engineering, and computer science.

 [Statistics], on the other hand, is the science of collecting, analyzing, and interpreting data. It is concerned with making inferences and drawing conclusions from data. Statistics provides methods for summarizing and describing data, as well as making predictions and testing hypotheses. It is used in many fields such as business, medicine, social sciences, and economics.

 In summary, probability is focused on the theoretical study of random events, while statistics is concerned with the practical application of data analysis to make inferences and draw conclusions.

 More at:
  * ...

 See also [P], ...


# Procedural Reasoning System
# PRS

 In artificial intelligence, a procedural reasoning system (PRS) is a framework for constructing real-time reasoning systems that can perform complex tasks in dynamic environments. It is based on the notion of a rational agent or intelligent agent using the belief–desire–intention software model. A user application is predominately defined, and provided to a PRS system is a set of knowledge areas. Each knowledge area is a piece of procedural knowledge that specifies how to do something, e.g., how to navigate down a corridor, or how to plan a path (in contrast with robotic architectures where the programmer just provides a model of what the states of the world are and how the agent's primitive actions affect them). Such a program, together with a PRS interpreter, is used to control the agent. The interpreter is responsible for maintaining beliefs about the world state, choosing which goals to attempt to achieve next, and choosing which knowledge area to apply in the current situation. How exactly these operations are performed might depend on domain-specific meta-level knowledge areas. Unlike traditional AI planning systems that generate a complete plan at the beginning, and replan if unexpected things happen, PRS interleaves planning and doing actions in the world. At any point, the system might only have a partially specified plan for the future. PRS is based on the BDI or [Belief-Desire-Intention Framework] for intelligent agents. Beliefs consist of what the agent believes to be true about the current state of the world, desires consist of the agent's goals, and intentions consist of the agent's current plans for achieving those goals. Furthermore, each of these three components is typically explicitly represented somewhere within the memory of the PRS agent at runtime, which is in contrast to purely reactive systems, such as the subsumption architecture.

 More at:
  * [https://indiaai.gov.in/article/understanding-procedural-reasoning-systems-in-ai](https://indiaai.gov.in/article/understanding-procedural-reasoning-systems-in-ai) 

 See also [P], ...


# Product Quantization
# PQ

 Product quantization is the process where each dataset vector is converted into a short memory-efficient representation (called PQ code). Instead of fully keeping all the vectors, their short representations are stored. At the same time, product quantization is a lossy-compression method which results in lower prediction accuracy but in practice, this algorithm works very well.

 More at:
  * [https://medium.com/@srivatssan/product-quantization-a2779ace565](https://medium.com/@srivatssan/product-quantization-a2779ace565)
  * [https://medium.com/towards-data-science/similarity-search-product-quantization-b2a1a6397701](https://medium.com/towards-data-science/similarity-search-product-quantization-b2a1a6397701)

 See also [P], ...


# PQ Code

 See also [P], [Product Quantization]


# Prompt Engineering

 ~ Acquire new capabilities at inference time

 **[Large Language Models] are like alien artifiacts that feel from the sky one day and we are still banging rocks against them trying to make then do something useful!**

 Impact of prompt engineering (aka prompt tuning!) on the large language model based on the [SuperGLUE Benchmark]

 ![]( {{site.assets}}/p/prompt_engineering_impact.png ){: width="100%"}

 Techniques:
  * [Zero-Shot Prompting]
  * [Few-Shot Prompting]
  * [Tree Of Thoughts (ToT) Prompting][ToT]
  * [Chain Of Thought (CoT) Prompting][CoT]
  * [Reason-Act (ReAct) Prompting][ReAct]
  * [Self-Consistency (SC) Prompting][SC]
  * [Step-Back Prompting] - ask a higher level question first, then answer the original question

 ![]( {{site.assets}}/p/prompt_engineering_techniques_comparison.png ){: width="100%"}

 ![]( {{site.assets}}/p/prompt_engineering_techniques_diagrams.png ){: width="100%"}

 More at:
  * Guide to prompt engineering - [https://www.promptingguide.ai/](https://www.promptingguide.ai/)
  * Promptbase to buy or sell prompts - [https://promptbase.com/](https://promptbase.com/)
  * Prompthub.us - [https://www.prompthub.us/](https://www.prompthub.us/)

 See also [P], [ChatGPT Model], [DALL-E Model]


# Prompt Injection

 More at:
  * example of PI - [https://twitter.com/goodside/status/1598253337400717313](https://twitter.com/goodside/status/1598253337400717313)

 See also [P], [ChatGPT Model], [GPT Model]


# Prompt Tuning

 See [Prompt Engineering]


# PromptIDE Application

 Integrated development environment (IDE) developed by [xAI] for prompt engineering and interpretability research

 The [xAI] PromptIDE is an integrated development environment for prompt engineering and interpretability research. It accelerates prompt engineering through an SDK that allows implementing complex prompting techniques and rich analytics that visualize the network's outputs. Used heavily in the development of [Grok].

 More at: 
  * [https://x.ai/prompt-ide/](https://x.ai/prompt-ide/)

 See also [P], ...


# Proximal Policy Optimization Algorithm
# PPO Algorithm

 We propose a new family of policy gradient methods for [reinforcement learning], which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

 {% youtube "https://www.youtube.com/watch?v=HrapVFNBN64" %}

 {% youtube "https://www.youtube.com/watch?v=5P7I-xPq8u8" %}

 {% youtube "https://www.youtube.com/watch?v=hlv79rcHws0" %}

 {% youtube "https://www.youtube.com/watch?v=KjWF8VIMGiY" %}

 {% pdf "{{site.assets}}/p/proximal_policy_optimization_paper.pdf" %}

 More at:
  * home - [https://openai.com/blog/openai-baselines-ppo/](https://openai.com/blog/openai-baselines-ppo/)
  * paper - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
  * code - [https://github.com/openai/baselines](https://github.com/openai/baselines)
  * articles
    * huggingface - [https://huggingface.co/blog/deep-rl-ppo](https://huggingface.co/blog/deep-rl-ppo)
    * pong from pixel - [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)

 See also [P], [Policy Gradient Algorithm], [Soft Actor-Critic Algorithm]


# Pruning

 Like other deep neural networks, large language models are composed of many components. However, not all of these components contribute significantly to the model’s output. In fact, some may have little to no effect at all. These non-essential components can be pruned, making the model more compact while maintaining the model’s performance.

 There are several ways to perform LLM / model pruning, each with its own set of advantages and challenges.
  * [Structured pruning]
  * [Unstructured pruning]

 See also [P], [Model Compression]


# Punishment

 In [Reinforcement Learning (RL)][RL], this is a negative [reward].

 More at:
  * ...

 See also [P], ...


# PyBullet

 {% youtube "https://www.youtube.com/watch?v=hmV4v_EnB0E" %}

 More at :
  * www - [https://pybullet.org/wordpress/](https://pybullet.org/wordpress/)
  * video paper - [https://xbpeng.github.io/projects/ASE/index.html](https://xbpeng.github.io/projects/ASE/index.html)

 See also [P], [Isaac Gym], [OpenAI Gym], [RobotSchool]


# Pycaret Python Module

 More at:
  * [https://pycaret.gitbook.io/docs/](https://pycaret.gitbook.io/docs/)

 See also [P], ...


# PyGame Python Module

 A [Python Module] that ...

 {% youtube "https://www.youtube.com/watch?v=PJl4iabBEz0" %}

 See also [P], [Gym Environment], [PyTorch ML Framework]


# Python Module

  * [Gradio] - to build a basic UI to interface with a model
  * [JAX] - 
  * [Joblib] - to save models in files
  * [LangChain] - LLMOps!
  * [Matplotlib] - for visualization
  * [Numpy] -
  * [Pandas] - to work with tabular data
  * [Pycaret] - A low-code machine learning library
  * [PyTorch] - A framework for deep learning
  * [PyTorch Geometric] - A framework for ML on graph
  * [Seaborn] - for visualization
  * [TensorFlow] - a framework for deep learning developed by [Google]

 Other modules
  * [Argparse] - take command line parameters
  * [PyGame] -

 See also [P], ...


# PyTorch Geometric Python Module

 Developed at Stanford

 {% youtube "https://www.youtube.com/watch?v=JAB_plj2rbA" %}

 More at:
  * docs - [https://pytorch-geometric.readthedocs.io/en/latest/index.html](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

 See also [P], ...


# PyTorch Python Module

 More at:
  * tutorials - [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

 See also [P], [Deep Learning Framework], [Machine Learning Framework]
