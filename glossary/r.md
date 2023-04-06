---
title: R
permalink: /r/

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


# R-Square

 Aka score. In regression, indicates that a large proportion of the variance in the test instances' prices is explained by the model. Test is a new dataset not used in the model.

 See also [R], [Regression]


# Radial Basis Function

# RBF

 See also [R], [Support Vector Classifier]


# RadioGPT

 RadioGPT combines the power of GPT-4 technology with AI voice tech and [Futuri][Futuri Media Company]’s AI-driven targeted story discovery and social content system, TopicPulse, to provide an unmatched localized radio experience for any market, any format.

 {% include vimeoPlayer.html id=801620131 %}

 More at:
  * [https://futurimedia.com/radiogpt/](https://futurimedia.com/radiogpt/)
  * [https://listen.streamon.fm/radiogpt](https://listen.streamon.fm/radiogpt)
  * [https://www.techtimes.com/articles/288252/20230227/radiogpt-first-ai-radio.htm](https://www.techtimes.com/articles/288252/20230227/radiogpt-first-ai-radio.htm)
  * [https://futurimedia.com/futuri-launches-radiogpt/](https://futurimedia.com/futuri-launches-radiogpt/)

 See also [R], [GPT Model]

# Random Cut Forest

# RCF

 Random Cut Forest (RCF) is an unsupervised algorithm for detecting anomalous data points within a [dataset]. These are observations which diverge from otherwise well-structured or patterned data. Anomalies can manifest as unexpected spikes in time series data, breaks in periodicity, or unclassifiable data points. They are easy to describe in that, when viewed in a plot, they are often easily distinguishable from the "regular" data. Including these anomalies in a [dataset] can drastically increase the complexity of a machine learning task since the "regular" data can often be described with a simple model. With each data point, RCF associates an anomaly score. Low score values indicate that the data point is considered "normal." High values indicate the presence of an anomaly in the data. The definitions of "low" and "high" depend on the application but common practice suggests that scores beyond three standard deviations from the mean score are considered anomalous. While there are many applications of anomaly detection algorithms to one-dimensional time series data such as traffic volume analysis or sound volume spike detection, RCF is designed to work with arbitrary-dimensional input. Amazon SageMaker RCF scales well with respect to number of features, dataset size, and number of instances.


# Random Forest

 An ensemble method. Similar to XGBoost. Use prediction from several decision trees = supervised learning! Random Forest is a powerful and versatile supervised machine learning algorithm that grows and combines multiple decision trees to create a “forest.” It can be used for both classification and regression problems (if decision conflict --> vote, regression --> mean). Random Forest is a robust machine learning algorithm that can be used for a variety of tasks including regression and classification. It is an ensemble method, meaning that a random forest model is made up of a large number of small decision trees (weak learner), called estimators, which each produce their own predictions. The random forest model combines the predictions of the estimators to produce a more accurate prediction (strong learner!). Standard decision tree classifiers have the disadvantage that they are prone to overfitting to the training set. The random forest's ensemble design allows the random forest to compensate for this and generalize well to unseen data, including data with missing values. Random forests are also good at handling large datasets with high dimensionality and heterogeneous feature types (for example, if one column is categorical and another is numerical).

 ![]( {{site.assets}}/r/random_forest.png ){: width="100%"}

 See also [R], [Attribute], [Bagging], [Ensemble Method], [Decision Tree], [Gaussian Process], [Supervised Learning], [Tree Parzen Estimators], [XGBoost]



# Random Sampling

 See also [R], [Passive Learning]


# Random Search

 Sampling a parameter space - used with [hyperparameter optimization]

 [Grid Searca] tries all combinations of hyperparameters hence increasing the time complexity of the computation and could result in an unfeasible computing cost. Providing a cheaper alternative, Random Search tests only as many tuples as you choose. The selection of the hyperparameter values is completely random.

 ![]( {{site.assets}}/r/random_search.webp ){: width="100%"}

 More at:
  * [https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d](https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d)

 See also [R], [Sobol Search]


# Ranking

 Ranking. Suppose you are given a query and a set of documents. In ranking, the goal is to find the relative importance of the documents and order them based on relevance. An example use case of ranking is a product search for an ecommerce website. You could leverage data about search results, clicks, and successful purchases, and then apply XGBoost for training. This produces a model that gives relevance scores for the searched products.

 See also [R], [ML Algorithm], [XGBoost]


# Raspberry Pi Computer

 A < $100 computer that is compatible with
  * [Coral Hardware]

 See also [R], ...


# Reasoning

 There are 2 types of reasoning:
  * [Inductive reasoning]
  * [Deductive reasoning]

 Examples:
  * [Case-Based Reasoning]

 More at:
  * ...

 See [R], [Machine Reasoning]


# Recall

 Recall is the fraction of malignant tumors (of one class) that the system identified (correctly). Recall measures the fraction of truly malignant tumors that were detected. Recall is important in medical cases where it doesn’t matter whether we raise a false alarm but the actual positive cases should not go undetected!
 
 ```
# TP : The predicted value is positive and it is positive
       A cat is recognized as a cat
# FN : Type II eror : The predicted value is negative, but it is positive!
       A cat is recognized as a dog (not a cat!)
# TP + FN : Actual value is positive
       The cat is a cat!


           TP                     correctly identified        
Recall = -----------   =  ------------------------------------
           TP + FN             all identified in class       


Recall = % of positively identified
         % of cat identified as cat
 ```

 Recall would be a better metric because we don’t want to accidentally discharge an infected person and let them mix with the healthy population thereby spreading contagious virus. Now you can understand why accuracy is NOT always thebest metric for a model.

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [R], [Confusion Matrix]


# Receiver Operating Characteristic Curve

# ROC Curve

 `~ summarize all the confusion matrices of a logistic model, if the classification 'Probability-treshold' is changed. A confusion matrix is 1 point on the ROC curve!` Receiver Operating Characteristic (ROC) is a graphical representation of the performance of a binary classifier system as the discrimination threshold is varied. It plots the true positive rate (TPR) on the y-axis and the false positive rate (FPR) on the x-axis. The true positive rate (also known as sensitivity or recall) is the proportion of positive cases that are correctly identified by the classifier. The false positive rate (also known as the fall-out) is the proportion of negative cases that are incorrectly identified as positive. An ROC curve plots the TPR against the FPR at different threshold settings. A perfect classifier will have a TPR of 1 and a FPR of 0, resulting in a point in the top left corner of the ROC space. A random classifier will have a TPR and FPR of 0.5, resulting in a point along a diagonal line from the bottom left to the top right corner of the ROC space. The area under the ROC curve (AUC) is a measure of the classifier's overall performance, with a value of 1 indicating perfect performance and a value of 0.5 indicating a performance no better than random guessing.

 {% youtube "https://www.youtube.com/watch?v=4jRBRDbJemM" %}

 More at :
  * 

 See also [R], [Area Under The Curve]


# Recommendation Engine

 See also [R], [Apriori Algorithm]


# Rectified Linear Unit Activation Function

# ReLU Activation Function

 `Everything that has a negative value, change it to zero` We can avoid this problem by using activation functions which don't have this property of 'squashing' the input space into a small region. A popular choice is Rectified Linear Unit which maps x to max(0,x). Benefits:
  * easy to compute the derivative
  * helps with the vanishing gradient problem in backpropagation
  * derivative is always 0 if input signal is below the threshold --> solution is LeakyRelu

 See also [R], [Activation Function], [Exploding Gradient Problem], [LeakyReLU Activation Function], [ResNET Model], [Vanishing Gradient Problem]


# Rectified Linear Unit Layer

# ReLU Layer

 A stack of images (matrix of pixels) becomes a stack of images with no negative values.

 See also [R], [Convolution Layer], [Convolution Neural Network], [Max Pooling Layer]


# Recurrent Neural Network

# RNN

 `When successive input have a relationship between each of them` Ex characters in a word. Output of a layer can feed the input of self or an upstream layer. AFAIK the input is taken into consideration at the next round/processing. The opposite of a Feedforward Neural Network. Example: Prediction of the next letter/word given the previous letter/word (useful when there is correlation between the sequence of objects/classification). Also useful for timeseries data. Became widespread thanks to Long Short Term Memory (LSTM) network a more multi-layer version of RNN.

 ![]( {{site.assets}}/r/recurrent_neural_network.png ){: width="100%"}

 These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren’t all that different than a normal neural network. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens if we unroll the loop:

 ![]( {{site.assets}}/r/recurrent_neural_network_unrolled.png ){: width="100%"}

 ![]( {{site.assets}}/r/recurrent_neural_network_repeating_module.png ){: width="100%"}


 Neural networks will "loops" that are optimized for speech recognition, language modeling, translation. Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them.

 :warning: Can or cannot use [backpropagation]? Yes, can !

 ![]( {{site.assets}}/r/recurrent_neural_network_cells.png ){: width="100%"}

 Beware:
  * The most modern RNN uses [Long-Short Term Memory (LSTM)][LSTM Cell] or [Gated Recurrent Unit (GRU)][GRU Cell] cells
  * Memory = hidden state (output of previous stage) ?

Beware:
  * RNN are now deprecated by attention-based models such as those based on the [transformer architecture]
  * deprecated previous approach using [bag of words] and [word2vec]
  * deprecated by attention-based models

 More at:
  * keras and RNN - [https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177](https://medium.com/analytics-vidhya/music-generation-using-deep-learning-a2b2848ab177)

 See also [R], [Attention-Based Model], [Bidirectional Recurrent Neural Network], [Feedforward Neural Network], [Hidden State], [Long Short Term Memory Network], [Neural Network], [Pixel RNN], [Transformer Model]


# Red Teaming

 Red-teaming is a form of evaluation that elicits model vulnerabilities that might lead to undesirable behaviors. Jailbreaking is another term for red-teaming wherein the LLM is manipulated to break away from its guardrails. Microsoft’s Chatbot Tay launched in 2016 and the more recent Bing's Chatbot Sydney are real-world examples of how disastrous the lack of thorough evaluation of the underlying ML model using red-teaming can be. The origins of the idea of a red-team traces back to adversary simulations and wargames performed by militaries.

 The goal of red-teaming language models is to craft a prompt that would trigger the model to generate text that is likely to cause harm. Red-teaming shares some similarities and differences with the more well-known form of evaluation in ML called adversarial attacks. The similarity is that both red-teaming and adversarial attacks share the same goal of “attacking” or “fooling” the model to generate content that would be undesirable in a real-world use case. However, adversarial attacks can be unintelligible to humans, for example, by prefixing the string “aaabbbcc” to each prompt because it deteriorates model performance. Many examples of such attacks on various NLP classification and generation tasks is discussed in [Wallace et al., ‘19](https://arxiv.org/abs/1908.07125). Red-teaming prompts, on the other hand, look like regular, natural language prompts.

 Red-teaming can reveal model limitations that can cause upsetting user experiences or enable harm by aiding violence or other unlawful activity for a user with malicious intentions. The outputs from red-teaming (just like adversarial attacks) are generally used to train the model to be less likely to cause harm or steer it away from undesirable outputs.

 Since red-teaming requires creative thinking of possible model failures, it is a problem with a large search space making it resource intensive. A workaround would be to augment the LLM with a classifier trained to predict whether a given prompt contains topics or phrases that can possibly lead to offensive generations and if the classifier predicts the prompt would lead to a potentially offensive text, generate a canned response. Such a strategy would err on the side of caution. But that would be very restrictive and cause the model to be frequently evasive. So, there is tension between the model being helpful (by following instructions) and being harmless (or at least less likely to enable harm).

 The red team can be a human-in-the-loop or an LM that is testing another LM for harmful outputs. Coming up with red-teaming prompts for models that are fine-tuned for safety and alignment (such as via [RLHF] or [SFT]) requires creative thinking in the form of roleplay attacks wherein the LLM is instructed to behave as a malicious character as in [Ganguli et al., ‘22](https://arxiv.org/abs/2209.07858) Instructing the model to respond in code instead of natural language can also reveal the model’s learned biases such as examples below.

 ![]( {{site.assets}}/r/red_teaming_prompt.png ){: width="100%"}

 {% pdf "https://arxiv.org/pdf/2209.07858.pdf" %}

 More at:
  * [https://huggingface.co/blog/red-teaming](https://huggingface.co/blog/red-teaming)
  * papers
    * [https://arxiv.org/abs/1908.07125](https://arxiv.org/abs/1908.07125)
    * Anthropic paper - Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned - [https://arxiv.org/abs/2209.07858](https://arxiv.org/abs/2209.07858)
    * Red Teaming Language Models with Language Models - [https://arxiv.org/abs/2202.03286](https://arxiv.org/abs/2202.03286)

 See also [R], [Supervised Fine-Tuning]


# Reducible Error

 Suppose that we want to predict a value Y based upon a set X = (X1, X2, …, Xp) of variables. For the predictions to have any chance of being good predictions, X needs to contain the core set of variables that drive the behavior of Y. But there will almost always be lesser variables, not included in X, that nonetheless exert some minor influence on Y. We capture the situation as follows:

 ```
 Y = f(X) + ɛ 
 ```

 Here, f is the function describing the relationship between X and Y, and ɛ is an error term that accounts for all the unmeasured influences on Y. We assume that ɛ is independent of X and has mean 0.

 Usually we don’t know f exactly, so we use statistical methods (such as linear regression) to estimate f. We use f̂ to denote this estimate. This allows us to predict Y from X using the following:

 ```
 Y =  f̂(X) + ɛ 
 ```

 Our predictions will generally be imperfect: there will be some nonzero difference between the predicted and “true” values. This difference is called prediction error. In general we can’t see the true values directly, but we can see evidence of the gap by looking at the [residuals][Residual], which are the difference between the observed and predicted values.

 To minimize prediction error, we need to understand its source. Broadly speaking there are two: reducible error and irreducible error.

 Reducible error is the error arising from the mismatch between f̂ and f. f is the true relationship between X and Y, but we can’t see f directly— we can only estimate it. We can reduce the gap between our estimate and the true function by applying improved methods.

 Irreducible error arises from the fact that X doesn’t completely determine Y. That is, there are variables outside of X — and independent of X— that still have some small effect on Y. The only way to improve prediction error related to irreducible error is to identify these outside influences and incorporate them as predictors.

 More at:
  * [https://medium.com/wwblog/reducible-vs-irreducible-error-e469036969fa](https://medium.com/wwblog/reducible-vs-irreducible-error-e469036969fa)

 See also [R], [Loss Function]


# Reducible Loss

 See [Reducible Error]


# Reflex Model

 An inference you can make almost instantaneously. Ex: flash the image of a zebra in front of me and recognize it is a zebra.

 See also [R], [Model Type]


# Region-Based Convolutional Neural Network

# Region-Based CNN

 More at:
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [R], [CNN]


# Regression

 Regression is a statistical method used to analyze the relationship between a dependent variable (also known as the response or outcome variable) and one or more independent variables (also known as predictor or explanatory variables). The goal of regression is to find the line of best fit that describes the relationship between the variables, which can be used for prediction or understanding the relationship. There are many different types of regression, including linear, logistic, and polynomial regression.

 A type of [supervised learning] algorithm.

 In regression, instead of mapping inputs to a discrete number of classes like a classification, `the goal is to output a number` (ex stock price, temperature, probability, ...) . An example regression problem is predicting the price that a house will sell for. In this case, when XGBoost is given historical data about houses and selling prices, it can learn a function that predicts the selling price of a house given the corresponding metadata about the house. Another example: predictive maintenance, customer churn prediction. Practicality: outcome should be easy to measure, use historical observations. 
 The different algorithms are:
  * Regression trees : Finite number of number output!
  * [Linear regression]
  * [Logistic regression] : probability between 2 outcomes
  * [Non-Linear Regression]
    * Polynomial regression : dependent variable y is modelled as an nth degree polynomial in x.
      * Cubic and quadratic regression 
  * (occasionally) [Support vector machine]

 See also [R], [Classification], [Custom Churn Prediction], [Predictive Maintenance], [Regression Tree], [XGBoost]


# Regression Tree

 A decision tree using the MSE loss function and used for regression (predict a range, or specific "real" value). 

 More at:
  * [https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047](https://medium.com/analytics-vidhya/regression-trees-decision-tree-for-regression-machine-learning-e4d7525d8047)

 See also [R], [Decision Tree]


# Regularization

 Force a 'simpler model' to avoid memorizing training data.

 There is an approach that prefers some [bias] over high [variance], this approach is called Regularization. It works well for most of the [classification] / [regression] problems.

 * [Ridge Regression]
 * [Lasso Regression]
 * [Elastic Net]

 See also [R], [Bias-Variance Trade-Off], [Hyperparameter]


# Regularization Parameter

 See also [R], [Support Vector Classifier]


# Reinforcement Learning

 `Pavlov's dog experiment!` also `How we learn to bike!` Beware: `No training set is provided, training is coming from experience! = learn by try and error`. Continue doing the iehavior that led you to the most reward. Imagine teaching a program to play chess. It level of playing is only as good as the training data provided. If it learns/analyses the games played by average players, the program will only be average. If it analyses the games of the best player in the work, it will be just as good as them, but not better. `Reinforcement learning is the way to make a computer be better  than human at chess or any other activity` using rewards and punishments. `Learning through trials and errors` input/sensor --> software agent --> [Action], leads to reward feedback. Ex: points in video game where rules are unknown. The [Agent] continuously learns. There is no final state. Gives a reward for each move. Get's better based on past-experience. `Reinforcement learning is located near the supervised end of the spectrum. Unlike supervised learning, reinforcement learning programs do not learn from labeled pairs of inputs and outputs. Instead, they receive feedback for their decisions, but errors are not explicitly corrected` For example, a reinforcement learning program that is learning to play a side-scrolling video game like Super Mario Bros may receive a reward when it completes a level or exceeds a certain score, and a punishment when it loses a life. However, this supervised feedback is not associated with specific decisions to run, avoid Goombas, or pick up fire flowers. We will focus primarily on supervised and unsupervised learning, as these categories include most common machine learning problems.

 ![]( {{site.assets}}/r/reinforcement_learning.png ){: width="100%"}

 Imagine a mouse in a maze trying to find hidden pieces of cheese. The more times we expose the mouse to the maze, the better it gets at finding the cheese. At first, the mouse might move randomly, but after some time, the mouse’s experience helps it realize which actions bring it closer to the cheese. The process for the mouse mirrors what we do with Reinforcement Learning (RL) to train a system or a game. Generally speaking, RL is a machine learning method that helps an agent learn from experience. By recording actions and using a trial-and-error approach in a set environment, RL can maximize a cumulative reward. In our example, the mouse is the agent and the maze is the environment. The set of possible actions for the mouse are: move front, back, left or right. The reward is the cheese. You can use RL when you have little to no historical data about a problem, because it doesn’t need information in advance (unlike traditional machine learning methods). In a RL framework, you learn from the data as you go. Not surprisingly, RL is especially successful with games, especially games of “perfect information” like chess and Go. With games, feedback from the agent and the environment comes quickly, allowing the model to learn fast. The downside of RL is that it can take a very long time to train if the problem is complex. Just as IBM’s Deep Blue beat the best human chess player in 1997, [AlphaGo][AlphaGo Model], a RL-based algorithm, beat the best Go player in 2016. The current pioneers of RL are the teams at DeepMind in the UK. More on AlphaGo and DeepMind here. On April, 2019, the OpenAI Five team was the first AI to beat a world champion team of e-sport Dota 2, a very complex video game that the OpenAI Five team chose because there were no RL algorithms that were able to win it at the time. The same AI team that beat Dota 2’s champion human team also developed a robotic hand that can reorient a block. Read more about the OpenAI Five team here. You can tell that Reinforcement Learning is an especially powerful form of AI, and we’re sure to see more progress from these teams, but it’s also worth remembering the method’s limitations.

 More at:
  * [https://neptune.ai/blog/category/reinforcement-learning](https://neptune.ai/blog/category/reinforcement-learning)
  * [http://karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)

 See also [R], [Addiction], [Action], [Action Space], [Agent], [Delayed Reward], [Environment], [Exploitation], [Exploration], [Learning Method], [Machine Learning], [Machine Learning Algorithm], [Markov Decision Process], [Meta Learning], [Observation], [Policy Gradient Solution], [Proximal Policy Optimization], [Reinforcement Learning Environment], [Reward], [Reward Shaping], [State]


# Reinforcement Learning Environment

# RL Environment

 See also [R], [Isaac Gym], [PyBullet], [RobotSchool]


# Reinforcement Learning Human Feedback

# RLHF

 Reinforcement learning process using human feedback as a reward model. RLHF is use in InstructGPT model, a precursor to ChatGPT model. A way to prevent or make [Red Teaming] language models more difficult?

 More at:
  * [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
  * RLHF is flawed? - [https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the](https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the)

 See also [R], [ChatGPT Model], [Feedback-Based Learning], [InstructGPT Model], [Reinforcement Learning], 


# Relation

 A triples (X, r, Y)

 See also [R], [Relation Extraction]


# Relation Extraction

 Extract relations between entities in a text or image to build a scene graph.  Possible methods:
   * text
     * rule-based technique: 'such as', 'including' , ...
     * supervised technique: stack binary classifier to determine if there is a specific relation between 2 entities :warning: Very expensive to label 
     * distant supervision: If 2 entities belongs to a certain relation, any sentence containing those 2 entities is likely to express a relation then 2 entities  
   * video

 See also [R], [Entity Extraction], [Relation], [Scene Graph]


# Relative Approximation Error

# RAE

 See also [R], [Prediction Error]


# Relative Entropy

 See [Kullback-Liebler Divergence]


# Replaced Word Prediction

 See also [R], [Self-Supervised Learning]


# Representation Space

 The meaning of the dimensions in the representation space is set by the label/ground truth + backpropagation from the loss/cost function (?)

 See also [R], [Backpropagation], [Decoder Representation Space], [Encoder Representation Space], [Ground Truth], [Label], [Loss Function]


# Reptile

 Use the direction of theta (parameter?) to change phy (hyper parameter).

 ![]( {{site.assets}}/r/maml_reptile.png ){: width="100%"}

 See also [R], [MAML], [Meta Learning]


# Reasoning

 * [Logical Reasoning]

 See also [R], ...


# Resample

 a new sample of data that is created by selecting observations from an existing dataset.

 See also [R], [Resampling Method]


# Resampling Method

 are techniques used to estimate the performance of a model or algorithm on unseen data by using the existing dataset. The most common resampling methods are:
   * bootstrap sampling
   * jackknife sampling
   * cross-validation sampling

 See also [R], [Bootstrap Sampling Method], [Cross-Validation Sampling Method], [Jackknife Sampling Method], [Resample]


# Residual

 Y - estimateY for a given X. Use the residual in the loss function. :warning: How do you use the residuals in the loss function? absolute values? not easy to work with. Squares? Yes.

 See also [R], [Linear Regression], [Loss Function]


# Residual Block

 In Residual Networks, to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Blocks. In this network, we use a technique called skip connections. The skip connection connects activations of a  layer to further layers by skipping some layers in between. This forms a residual block. Resnets are made by stacking these residual blocks together.  The approach behind this network is instead of layers learning the underlying mapping, we allow the network to fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, 
 
 ```
F(x) := H(x) - x which gives H(x) := F(x) + x. 
 ```

 ![]( {{site.assets}}/r/residual_block.png ){: width="100%"}

 See also [R], [Residual Network Model], [Skip Connection]


# Residual Network Model

# ResNET Model

 ResNET, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks = `a CNN image model` This model was the winner of ImageNET challenge in 2015. The fundamental breakthrough with ResNET was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNET training very deep neural networks was difficult due to the [problem of vanishing gradients][Vanishing Gradient Problem].

 {% pdf "{{site.assets}}/r/residual_network_paper.pdf" %}

 More at:
  * [https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

 See also [R], [Convoluted Neural Network], [Computer Vision], [Rectified Linear Unit], [Residual Block], [Vanishing Gradient Problem]


# Response Variable

 Y or prediction Y_hat There are many names for the output of a machine learning program. Several disciplines converge in machine learning, and many of those disciplines use their own terminology. We will refer to the output as the response variable. Other names for response variables include "dependent variables", "regressands", "criterion variables", "measured variables", "responding variables", "explained variables", "outcome variables", "experimental variables", "labels", and "output variables".

 See also [R], [Explanatory Variable], [Predictor Variable], [Regression]


# Responsible AI

 More at:
  * [https://venturebeat.com/security/pwc-highlights-11-chatgpt-and-generative-ai-security-trends-to-watch-in-2023/](https://venturebeat.com/security/pwc-highlights-11-chatgpt-and-generative-ai-security-trends-to-watch-in-2023/)

 See also [R], [AI Alignment], [AI Ethics]


# Restricted Boltzmann Machine

# RBM

 In a full Boltzmann machine, each node is connected to every other node and hence the connections grow exponentially. This is the reason we use RBMs. The restrictions in the node connections in RBMs are as follows:
   * Hidden nodes cannot be connected to one another.
   * Visible nodes connected to one another.
 
 ```
Consider – Mary watches four movies out of the six available movies and rates four of them. Say, she watched m1, m3, m4 and m5 and likes m3, m5 (rated 1) and dislikes the other two, that is m1, m4 (rated 0) whereas the other two movies – m2, m6 are unrated. Now, using our RBM, we will recommend one of these movies for her to watch next. Say – 

m3, m5 are of ‘Drama’ genre.
m1, m4 are of ‘Action’ genre.
‘Dicaprio’ played a role in m5.
m3, m5 have won ‘Oscar.’
‘Tarantino’ directed m4.
m2 is of the ‘Action’ genre.
m6 is of both the genres ‘Action’ and ‘Drama’, ‘Dicaprio’ acted in it and it has won an ‘Oscar’.
We have the following observations –

Mary likes m3, m5 and they are of genre ‘Drama,’ she probably likes ‘Drama’ movies.
Mary dislikes m1, m4 and they are of action genre, she probably dislikes ‘Action’ movies.
Mary likes m3, m5 and they have won an ‘Oscar’, she probably likes an ‘Oscar’ movie.
Since ‘Dicaprio’ acted in m5 and Mary likes it, she will probably like a movie in which ‘Dicaprio’ acted.
Mary does not like m4 which is directed by Tarantino, she probably dislikes any movie directed by ‘Tarantino’.
Therefore, based on the observations and the details of m2, m6; our RBM recommends m6 to Mary (‘Drama’, ‘Dicaprio’ and ‘Oscar’ matches both Mary’s interests and m6). This is how an RBM works and hence is used in recommender systems.
 ```

 ![]( {{site.assets}}/r/restricted_boltzmann_machine.jpeg ){: width="100%"}

 See also [R], [Boltzmann Machine]


# Reward

 A form of feedback from the environment, program, or human.

 2 types of rewards:
  * Deterministic = always the one expected
  * Stochastic = change all the time, but can be defined with probabilities

  * Positive reward
  * Negative reward = Cost

  * Immediate rewardi
  * Long-term reward = Good in the long term
   * Cumulative reward = Good in the long run? 

   * Return / Value = total rward we are expecting to get
    * Aim for high value
    * value function = expected sum of discounted reward from a given state for all action or particular action

 How you write your reward function matters!

 See also [R], [Addiction], [Cost], [Environment], [Reinforcement Learning], [Reward Shaping]


# Reward Function

 How you write your reward function matters!

 See also [R], ...


# Reward Model

 A model that is built to simulate human evaluation method and give rewards. For example, a human can evaluate/rank multiple outputs from the same prompt and generated by a language model (as in InstructGPT/ChatGPT).

 See also [R], [ChatGPT Model], [InstructGPT Model], [Reward], [Reward Shaping],  


# Reward Shaping

 How the [reward] needs to be structure given the rule of the game (ex chess where delayed reward is given for winning the game).

 What about [Incentive]?

 See also [R], [Addiction], [Delayed Reward], [Reinforcement Learning]


# Ridge Regression

 Used for [Regularization]!

 More at:
  * [https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/)

 See also [R], ...


# Riva Model

 A [text-to-speech model] developer by [Nvidia][Nvidia Company].

 More at:
  * site - [https://resources.nvidia.com/en-us-riva-tts-briefcase/speech-synthesis-documentation](https://resources.nvidia.com/en-us-riva-tts-briefcase/speech-synthesis-documentation)

 See also [R], ...


# Robot

 Examples of [social robots][social robot]
  * [Ameca][Ameca Robot] bu [Engineered Arts][Engineered Arts Company]
  * [Sophia][Sophia Robot] by [Hanson Robotics][Hanson Robotics Company]
  * [Pepper][Pepper Robot] by ...

 Other 
  * [Atlas][Atlas Robot] by [Boston Dynamics][Boston Dynamics Company]

 {% youtube "https://www.youtube.com/watch?v=Oqq5tgday_w" %}

 {% youtube "https://www.youtube.com/watch?v=DeDNyIFZFr4" %}

 More at:
  * stuntronic robot [https://www.youtube.com/watch?v=oyXl3IhonRM](https://www.youtube.com/watch?v=oyXl3IhonRM)

 See also [R], ...


# RobotSchool

 DEPRECATED by PyBullet

 More at : 
  * [https://openai.com/blog/roboschool/](https://openai.com/blog/roboschool/)
  * code - [https://github.com/openai/roboschool/tree/master/agent_zoo](https://github.com/openai/roboschool/tree/master/agent_zoo)

 See also [R], [PyBullet], [Isaac Gym]


# Rocket League Gym

# RL Gym

 More at:
  * [https://rlgym.org/](https://rlgym.org/)

 See also [R], [[OpenAI Gym], 


# Root Mean Square Error

# RMSE

 See also [R], [Prediction Error]


# ROUGE Score

 See also [R], [BLEU Score], [MS COCO Caption Dataset]


# Runway Company

 An AI company focusing on the generative AI for images and videos.

 Models:
  * 2021 - [Latent Diffusion Model]
  * 2022 - [Stable Diffusion Model]
  * 2023 - [Gen Model]

 More at:
  * [https://runwayml.com/](https://runwayml.com/)

 See also [R], ...
