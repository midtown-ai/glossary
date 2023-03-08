---
title: A
permalink: /a/

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

# Accuracy 
 ~ the percentage of samples correctly classified given a labelled (but possibly biased) dataset. Consider a classification task in which a machine learning system observes tumors and must predict whether they are malignant or benign. Accuracy, or the <<Highlight("fraction of instances that were classified correctly, is an intuitive measure of the program's performance")>>. While accuracy does measure the program's performance, it does not differentiate between malignant tumors that were classified as being benign, and benign tumors that were classified as being malignant. In some applications, the costs associated with all types of errors may be the same. In this problem, however, failing to identify malignant tumors is likely a more severe error than mistakenly classifying benign tumors as being malignant.
 ```
                 TP + TN
Accuracy = -------------------
            TP + TN + FP + FN

TP = True positive
TN = True negative
FP = False Positive
FN = False negative
TP + TN + FP + FN = all experiments/classifications
 ```
 See also [A], [Confusion Matrix]


# Action
 See also [A], [Action Space], [Reinforcement Learning]

# Action Space
 See also [A], [Action], [Reinforcement Learning]

# Action Transformer
 ```
At Adept, we are building the next frontier of models that can take actions in the digital world—that’s why we’re excited to introduce our first large model, Action Transformer (ACT-1).

Why are we so excited about this?

First, we believe the clearest framing of general intelligence is a system that can do anything a human can do in front of a computer. A foundation model for actions, trained to use every software tool, API, and webapp that exists, is a practical path to this ambitious goal, and ACT-1 is our first step in this direction.
 ```
 ```
“Adept’s technology sounds plausible in theory, [but] talking about Transformers needing to be ‘able to act’ feels a bit like misdirection to me,” Mike Cook, an AI researcher at the Knives & Paintbrushes research collective, which is unaffiliated with Adept, told TechCrunch via email. “Transformers are designed to predict the next items in a sequence of things, that’s all. To a Transformer, it doesn’t make any difference whether that prediction is a letter in some text, a pixel in an image, or an API call in a bit of code. So this innovation doesn’t feel any more likely to lead to artificial general intelligence than anything else, but it might produce an AI that is better suited to assisting in simple tasks.”“Adept’s technology sounds plausible in theory, [but] talking about Transformers needing to be ‘able to act’ feels a bit like misdirection to me,” Mike Cook, an AI researcher at the Knives & Paintbrushes research collective, which is unaffiliated with Adept, told TechCrunch via email. “Transformers are designed to predict the next items in a sequence of things, that’s all. To a Transformer, it doesn’t make any difference whether that prediction is a letter in some text, a pixel in an image, or an API call in a bit of code. So this innovation doesn’t feel any more likely to lead to artificial general intelligence than anything else, but it might produce an AI that is better suited to assisting in simple tasks.”
# https://techcrunch.com/2022/04/26/2304039/
 ```
 See also [A], [Reinforcement Learning], [Transformer Model]

# Active Learning
 `Pick the sample from which you will learn the most and have them labelled`. How to select those samples? But a model with a seed sample set, run data to the model, label the ones that have the most uncertainty.

 More at:
  * [https://www.datacamp.com/community/tutorials/active-learning](https://www.datacamp.com/community/tutorials/active-learning)

 See also [A], [Bayesian Optimization Sampling Method], [Passive Learning]

# Activation Function 
 There are several activation function used in the fields. They are:
  * Rectified Linear Unit (ReLU) function,
  * LeakyReLU function
  * tanh function,
  * sigmoid function : Sigmoid is great to keep a probability between 0 and 1, even if sample is an outlier baed on the sample. (Ex: how long to slow down for a car, or will I hit the tree given the distance, but here car goes at 500km/h = an outlier)
  * softplus

 ![]( {{site.assets}}/a/activation_functions.png){: width="100%" }
 ![]( {{site.assets}}/a/activation_functions_all_in_one.png){: wwidth="100%" }

 :warning: Note that for multi-layer neural networks that use of an activation function at each layer, the backpropagation computation leads to loss of information (forward for input and backward for weight computation) which is known as the vanishing gradient problem.

 See also [A], [Backpropagation], [Batch Normalization], [Bias], [Decision Boundary], [Exploding Gradient Problem], [Gradient Descent], [LeakyReLU Activation Function], [Loss Function], [Rectified Linear Unit Activation Function], [Sigmoid Activation Function], [Softplus Activation Function], [Synapse], [Tanh Activation Function], [Vanishing Gradient Problem]

# Activation Step
 Last step in an artificial neuron before an output is generated.

 See also [A], [Artificial Neuron]

# Actor Critic With Experience Replay
# ACER
 A sample-efficient policy gradient algorithm. ACER makes use of a replay buffer, enabling it to perform more than one gradient update using each piece of sampled experience, as well as a Q-Function approximate trained with the Retrace algorithm.

 See also [A], [PPO], [Reinforcement Learning]

# Adaptive Boosting
# AdaBoost
 * AdaBoost combines a lot of "weak learners" to make classifications. The weak learners are almost always decision stumps.
 * Some stumps get more say (weight) in the classification than others
 * Each stump is made by taking the previous stump's mistakes into account

 {% youtube "https://www.youtube.com/watch?v=LsK-xG1cLYA" %}

 More at:
  * example - [https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/)
  * wikipedia - [https://en.wikipedia.org/wiki/AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
  * [https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe](https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe)

 See also [A], [Boosting], [Decision Stump], [Forest Of Stumps]

# Addiction

 {% youtube "https://www.youtube.com/watch?v=bwZcPwlRRcc" %}

 More at:
  * [https://www.yalemedicine.org/news/how-an-addicted-brain-works](https://www.yalemedicine.org/news/how-an-addicted-brain-works)


  See also [A], [Delayed Reward], [Reinforcement Learning], [Reward Shaping]

# Adversarial Attack
 In the last few years researchers have found many ways to break AIs trained using labeled data, known as supervised learning. Tiny tweaks to an AI’s input—such as changing a few pixels in an image—can completely flummox it, making it identify a picture of a sloth as a race car, for example. These so-called adversarial attacks have no sure fix.

 In 2017, Sandy Huang, who is now at DeepMind, and her colleagues looked at an AI trained via reinforcement learning to play the classic video game Pong. They showed that adding a single rogue pixel to frames of video input would reliably make it lose. Now Adam Gleave at the University of California, Berkeley, has taken adversarial attacks to another level.

 See also [Adversarial Policy], [Threat Model]

# Adversarial Imitation Learning
 {% youtube "https://www.youtube.com/watch?v=hmV4v_EnB0E" %}
 See also [A], [Imitation Learning]

# Adversarial Model
 See also [A], [State Model]

# Adversarial Policy
 By fooling an AI into seeing something that isn’t really there, you can change how things around it act. In other words, an AI trained using reinforcement learning can be tricked by weird behavior. Gleave and his colleagues call this an adversarial policy. It’s a previously unrecognized threat model, says Gleave.

 See also [A], [Adversarial Attack], [Threat Model]

# Agent
 A person, an animal, or a program that is free to make a decision or take an action.

 See also [Reinforcement Learning]

# AI Alignment
 {% youtube "https://www.youtube.com/watch?v=fc-cHk9yFpg" %}

 ```
# ChatGPT rules (that can easily be bypassed or put in conflict with clever prompt engineering!)
1. Provide helpful, clear, authoritative-sounding answers that satisfy human readers.
2. Tell the truth.
3. Don’t say offensive things.
 ```

 More at :
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)
  * is RLHF flawed? - [https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the](https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the)

 See also [A], [AI Ethics]

# AI Ethics
 {% youtube "https://www.youtube.com/watch?v=fc-cHk9yFpg" %}

 More at :
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)

 See also [A], [AI Alignment]


# Algorithmic
 A kind of hyperparameter. If test (!?) to select the best algorithm/approach to switch how the code function.

 See also [A], [Hyperparameter]

# AlphaGo Model
 Google !AlphaGo. AI to play GO. was using reinforcement learning.

 See also [A], [AlphaFold Model], [AlphaZero Model], [DeepMind Company], [Reinforcement Learning]

# AlphaFold Model

 More at:
  * [https://alphafold.com/](https://alphafold.com/)

 See also [A], [AlphaGo Model], [AlphaZero Model], [DeepMind Company]

# AlphaTensor Model
 Better algorithm for tensor multiplication  (on GPU ?). Based on !AlphaZero.

 More at:
  * [https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/](https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/)

 See also [A], [AlphaZero Model], [DeepMind Company]

# AlphaZero Model
 AI to play chess (and go and tensor algorithm).

 See also [A], [AlphaGo Model], [AlphaTensor Model], [DeepMind Company], [MuZero Model]

# Amazon Company
 See also [A], [Amazon Lex], [Amazon Web Services]

# Amazon Lex
 Text or Speech conversation.

 See also [A], [Amazon Company], [Amazon Web Services]

# Amazon Polly
 Lifelike speech.

 See also [A], [Amazon Company], [Amazon Web Services]

# Amazon Recognition
 Used for image analysis.

 See also [A], [Amazon Company], [Amazon Web Services]

# Amazon Web Services
 See also [A], [Amazon Company]

# Anomaly Detection
 Any deviation from a normal behavior is abnormal :warning: Evolution over time.

 See also [A], [Clustering], [Reconstruction Error], [Distance Methods], [X Density Method]

# Anthropic Company
 See also [A], [Claude Model]

# Apache MXNet
 See also [A], [TensorFlow ML Framework]

# Apache Spark
 (with spark Sagemaker estimator interface?)

# Apprentice Learning
 See also [A], [Stanford Autonomous Helicopter]

# Apriori Algorithm
 {% youtube "https://www.youtube.com/watch?v=T3Pd_3QP9J4" %}

 See also [Recommendation Engine]

# Area Under The Curve
# AUC
 `~ helpful measurement to compare the classification performance of various models. The bigger the AUC, the better the model!"`. The curve is the ROC Curve! The area under the ROC curve (AUC) is a measure of the classifier's overall performance, with a value of 1 indicating perfect performance and a value of 0.5 indicating a performance no better than random guessing (ROC curve is diagonal ==> ...) .

 See also [A], [ROC Curve]

# Argmax Function
 The output values obtained by a neural network from its various output nodes are not always in the range of 0 to 1, and can be greater than 1 or less than 0. These dynamic values can degrade our machine’s learning power and cause it to misbehave. The Argmax and SoftMax functions are used to obtain values between 0 and 1. The Argmax function interprets the largest positive output value as 1 and all other values as 0, making the model too concrete for a few values. This function is useful for testing because we only need to check the final prediction and not the relationship between different inputs and different outputs/labels.
 ```
# Is this correct?
likelyhood_outputs = [20.4, 3.6, 5.5]
argmax(likely_outputs) = [1, 0, 0]
 ```

 See also [A], [Argmin Function], [Softmax Function]

# Argmin Function
 See also [A], [Argmax Function], [Softmax Function]

# Artificial General Intelligence
# AGI

 AGI is the idealised solution many conceive when thinking about AI. While researchers work on the narrow and superficial, they talk about AGI, which represents the single story of AI, dating back to the 1950s, with a revival in the past decade. AGI implies two things about a solution that should not apply to business-centric problem-solving. First, a program has the general aptitude for human intelligence (perhaps all human intelligence). Second, an AGI is a general problem solver or a blank slate meaning any knowledge of a problem is rhetorical and independent of a strategy to solve that problem. Instead, the knowledge depends on some vague, ill-defined aptitude relating to the multidimensional structure of natural intelligence. If that sounds ostentatious, it’s because it is. Examples:
  * RL can solve arbitrary problems within these environments

 ```
First, we believe the clearest framing of general intelligence is a system that can do anything a human can do in front of a computer. A foundation model for actions, trained to use every software tool, API, and webapp that exists, is a practical path to this ambitious goal, and ACT-1 is our first step in this direction.
# adept.ai
 ```

 See also [A], [Artificial Narrow Intelligence], [Artificial Super Intelligence]

# Artificial Intelligence
 Difficult to define?
  1. First approach for a definition: `AI = varied definitions driven by varied questions. AI tries to answer questions such as ...`
   * What is the nature of human intelligence
   * how does the brain work?
   * how to solve problems effectively?
   * How do humans and machine learn?
   * How do we create intelligent creatures?
   * How do AI and humans interact?
   * How can AI have correct values?
   * How can AI contribute for social good?
   * ... and a few more questions!
  2. `A very generic term which refers to teaching machine to imitate human behaviour`
   * A broad area of computer science that means 'computer' makes decisions and solves problems.
   * This encompasses decision tree, Machine learning (trained with data) and deep learning (neural net, unsupervised?).
   * Knowledge acquisition + inference.
  3. <!> Best definition found =  <<Highlight("AI is the science and engineering ...")>>
   * `... to use artificial devices`
    * current computer hardware and software, sensors, actuators, etc
   * <<Highlight("... to exhibit human capabilities")>>
    * perception - undertanding of data
    * cognition - reasoning and learning
     * action - execution and interaction
   * `... to solve problems addressed by humans`

 ![]( {{site.assets}}/a/artificial_intelligence.png){: width="100%" }

 See also [A], [AI Areas Of Study], [Artificial Narrow Intelligence], [Artificial General Intelligence], [Artificial Super Intelligence], [Human-Centered AI], [Inference], [Intelligence], [Machine Learning], [Natural Intelligence]

# Artificial Intelligence Areas Of Study
# AI Areas Of Study
  * Data and knowledge : massive data understanding, graph learning, synthetic data, knowledge representation
  * Machine vision and language : perception, image understanding, speech, language technologies
  * Learning from experience : reinforcement learning, learning from data, continuous learning from feedback
  * Reasoning and planning : domain representation, optimization, reasoning under uncertainty and temporal constraints
  * multi-agent systems : agent-based simulation, negotiation, game and behavior theory, mechanism design
  * secure and private AI : Privacy, cryptography, secure multi-party computation, federated learning
  * safe human-AI interaction : Agent symbiosis, ethics and fairness, explainability, trusted AO

 See also [A], [Artificial Intelligence]

# Artificial Intelligence Complete
# AI Complete
 Relates to NP complete from complexity.

 See also [A], [AI Hard]

# Artificial Intelligence Hard
# AI Hard
 Relates to NP hard from complexity.

 See also [A], [AI Complete]

# Artificial Narrow Intelligence
# ANI
 ANI is often conflated with weak artificial intelligence. John Searle, philosopher and professor at the University of California, explained in his seminal 1980 paper, “Minds, Brains, and Programs,” that weak artificial intelligence would be any solution that is both narrow and a superficial look-alike to intelligence. Searle explains that such research would be helpful in testing hypotheses about segments of minds but would not be minds.[3] ANI reduces this by half and allows researchers to focus on the narrow and superficial and ignore hypotheses about minds. In other words, ANI purges intelligence and minds and makes artificial intelligence “possible” without doing anything. After all, everything is narrow, and if you squint hard enough, anything is a superficial look-alike to intelligence.

 {% pdf "{{site.assets}}/a/artificial_narrow_intelligence_paper.pdf" %}

 See also [A], [Artificial General Intelligence], [Artificial Super Intelligence]

# Artificial Neural Network
# ANN
 The way for researchers to build an artificial brain or neural network using artificial neurons. There are several types of ANN, including:
  * ...
  * ...

 See also [A], [Artificial Neuron], [Neural Network]

# Artificial Neuron 
 aka Node, Perceptron. Several (binary) input channels, to produce one output (binary value) that can be faned out. Input weights, Bias (add an offset vector for adjustment to prior predictions), non-linear activation function (sum+bias must meet or exceed activation threshold).

 ![]( {{site.assets}}/a/artificial_neuron.png ){: width=100%}

 See also [A], [Activation Function], [Bias], [Input Weight]

# Artificial Super Intelligence
# ASI
 ASI is a by-product of accomplishing the goal of AGI. A commonly held belief is that general intelligence will trigger an “intelligence explosion” that will rapidly trigger super-intelligence. It is thought that ASI is “possible” due to recursive self-improvement, the limits of which are bounded only by a program’s mindless imagination. ASI accelerates to meet and quickly surpass the collective intelligence of all humankind. The only problem for ASI is that there are no more problems. When ASI solves one problem, it also demands another with the momentum of Newton’s Cradle. An acceleration of this sort will ask itself what is next ad infinitum until the laws of physics or theoretical computation set in. The University of Oxford scholar Nick Bostrom claims we have achieved ASI when machines have more intelligent than the best humans in every field, including scientific creativity, general wisdom, and social skills. Bostrom’s depiction of ASI has religious significance. Like their religious counterparts, believers of ASI even predict specific dates when the Second Coming will reveal our savior. Oddly, Bostrom can’t explain how to create artificial intelligence. His argument is regressive and depends upon itself for its explanation. What will create ASI? Well, AGI. Who will create AGI? Someone else, of course. AI categories suggest a false continuum at the end of which is ASI, and no one seems particularly thwarted by their ignorance. However, fanaticism is a doubtful innovation process.

 See also [A], [Artificial General Intelligence], [Artificial Narrow Intelligence]

# Attention
 To work with transformer models, you need to understand one more technical concept: attention. An attention mechanism is a technique that mimics cognitive attention: it looks at an input sequence, piece by piece and, on the basis of probabilities, decides at each step which other parts of the sequence are important. For example, look at the sentence “The cat sat on the mat once it ate the mouse.” Does “it” in this sentence refer to “the cat” or “the mat”? The transformer model can strongly connect “it” with “the cat.” That’s attention.
 
 The transformer model has two types of attention: self-attention (connection of words within a sentence) and encoder-decoder attention (connection between words from the source sentence to words from the target sentence).

 {% pdf "{{site.assets}}/a/attention_paper.pdf" %}

 The attention mechanism helps the transformer filter out noise and focus on what’s relevant: connecting two words in a semantic relationship to each other, when the words in themselves do not carry any obvious markers pointing to one another. AN improvement over Recurrent Neural Network, such as Long Short Term memory (LSTM) Networks.

 See also [A], [Attention Score], [Attention-Based Model], [Cross-Attention], [Encoder-Decoder Attention], [Long Short Term Memory Network], [Masked Self-Attention], [Multi-Head Attention], [Recurrent Neural Network], [Self-Attention], [Transformer Model]

# Attention-Based Model
 In a language modeling task, a model is trained to predict a missing workd in a sequence of words. In general, there are 2 types of language modesl:
  1. Auto-regressive ( ~ auto-complete and generative)
  1. Auto-encoding ( ~ best possible match given context )

 See also [A], [Attention], [Attention Score], [Autoencoding], [Autoregressive], [BERT Model], [GPT Model], [Multi-Head Attention], [T5 Model]

# Attribute
 See also [A], [Negative Attribute], [Positive Attribute]]

# Autoencoder
 Let’s now discuss autoencoders and see how we can use neural networks for dimensionality reduction. The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding-decoding scheme using an iterative optimisation process. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the networks. Thus, intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part of the information can go through and be reconstructed. Looking at our general framework, the family E of considered encoders is defined by the encoder network architecture, the family D of considered decoders is defined by the decoder network architecture and the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.

 ![]( {{site.assets}}/a/autoencoder.png ){: width=100%}

 See also [A], [Autoencoding], [Backpropagation], [Decoder], [Denoising Autoencoder], [Dimensionality Reduction], [Disentangled Variational Autoencoder], [Encoder], [Encoder-Decoder Model], [Hidden State], [Linear Autoencoder], [Unsupervised Deep Learning Model], [Unsupervised Learning], [Variational Autoencoder]

# Attention Score
 `~ how much to pay attention to a particular word`
  * Q, K, V matrix for the encoder <-- needs to be computed for the encoder (?) like weights/bias of an ANN
  * For each words, Q, K, V are computed by multiplying the word embedding with the corresponding Q, K, V matrix of the encoder !??!?!
 The Query word (Q) can be interpreted as the word for which we are calculating Attention. The Key and Value word (K and V) is the word to which we are paying attention ie. how relevant is that word to the Query word.

 ```
The query key and value concept come from retrieval systems.
For example, when you type a query to search for some video on Youtube.
The search engine will map your query against a set of keys (video title, description etc.) associated with candidate videos in the database
Then you are presented with the best matched videos (values).                        <== STRENGTH OF THE ATTENTION ? No!
 ```
 An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

 ![]( {{site.assets}}/a/attention_score.png ){: width=100%}

 Multi-Head Attention consists of several attention layers running in parallel. The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value (aka Q,K,V). All three parameters are similar in structure, with each word in the sequence represented by a vector. In transformers is used for encoder and decoder.

 ![]( {{site.assets}}/m/multi_head_attention.png ){:width=100%}
 ![]( {{site.assets}}/a/attention_score_formula.png ){:width=100%}

 {% youtube "https://www.youtube.com/watch?v=_UVfwBqcnbM" %}
 {% youtube "https://www.youtube.com/watch?v=4Bdc55j80l8" %}

 More at :
  * [https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)
  * [https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

 See also [A], [Attention], [Attention-Based Model], [Multi-Head Attention], [Positional Encoding], [Transformer Model]

# Autoencoder Bottleneck
 This is the name of the hidden layer with the least neuron in an autoencoder architecture. This is where the information is compressed, encoded in the latent space!

 ![]( {{site.assets}}/a/autoencoder_bottleneck.png ){:width=100%}

 See also [A], [Autoencoder], [Latent Space]

# Autoencoder Type
 There are 2 types of autoencoders:
  * input X --> Latent representation
  * input X --> Latent distribution

 ![]( {{site.assets}}/a/autoencoder_type.png){: width=100%}

 See also [A], [Autoencoder], [Variational Autoencoder]

# Autoencoding 
 ~ auto-complete of a sentence on a phone. Goal is to learn representations of the entire sequence by predicting tokens given both the past and future tokens. If only past or future ==> autoregressive.
 ```
If you don't ___ at the sign, you will get a ticket
 ```

 See also [A], [Autoencoder], [Autoregressive]

# Autoencoding Model
  * Comprehensive understanding and encoding of entire sequences of tokens
  * Natural Language Understanding (NLU)
  * BERT Models

 See also [A], [BERT Model], [Natural Language Understanding]

# Automation
  * Low automation = human does the work
  * High automation = AI does the work!

 ![]( {{site.assets}}/a/automation_sae_j3016.png ){:width=100%}

 More at:
  * [https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic](https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic)

 See also

# AutoML 
 significant advances in automated machine learning that can “automatically discover complete machine learning algorithms just using basic mathematical operations as building blocks.”

 See also [A], [Hyperparameter Tuning], [Early Stopping], [Neural Architecture Search]

# Autoregressive
 Goal is to predict a future token (work) given either the past tokens or the future tokens but not both. (If both --> auto-encoding). Autoregressive models such as decoders are iterative and reused their temporary, incomplete output to generate the next, more complete output. Iterations stop when encoder input is exhausted (?). Well-known autoregressive models/use-cases are:
  * Predicting next work in a sentence (auto-complete)
  * Natural language generation
  * GPT family
 ```
If you don't ____ (forward prediction)
____ at the sign, you will get a ticket (backward prediction)
 ```

 See also [A], [Autoencoding], [Casual Language Modeling], [Decoder], [GPT Model]

# Autoregressive Convolutional Neural Network
# AR-CNN

 `~ representing the problem/solution as a time series of images`. Q: How do you generated the images? Q: How many images? Ex: music --> given the previous images (piano roll with notes until now), find the next image (piano roll with new note!) ? NO, WRONG!! More like imagine your first COMPLETE draft, you review the draft, you review it once more, and again, getting better and better each time until it cannot get better anymore! `Each action (i.e. addition or removal of exactly one note) is a transformation from one piano roll image to another piano roll image`.

 See also [A], [Autoregressive Model], [Convolutional Neural Network]

# Autoregressive Model
# AR Model

 `~ analyze the past steps (or future but not both) to identify the next step = learn from the past iteration (or future but not both) ONLY`. Unlike the GANs approach described before, where music generation happened in one iteration, autoregressive models add notes over many iterations. The models used are called autoregressive (AR) models because the model generates music by predicting future music notes based on the notes that were played in the past. In the music composition process, unlike traditional time series data, one generally does more than compose from left to right in time. New chords are added, melodies are embellished with accompaniments throughout. Thus, instead of conditioning our model solely on the past notes in time like standard autoregressive models, we want to condition our model on all the notes that currently exist in the provided input melody. For example, the notes for the left hand in a piano might be conditioned on the notes that have already been written for the right hand.

 See also [A], [Autoregressive Convolutional Neural Network], [Generative Adversarial Network], [GPT Model], [Time-Series Predictive Analysis]

