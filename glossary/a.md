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

 ~ the percentage of samples correctly classified given a labelled (but possibly biased) dataset. Consider a classification task in which a machine learning system observes tumors and must predict whether they are malignant or benign. Accuracy, or the `fraction of instances that were classified correctly, is an intuitive measure of the program's performance`. While accuracy does measure the program's performance, it does not differentiate between malignant tumors that were classified as being benign, and benign tumors that were classified as being malignant. In some applications, the costs associated with all types of errors may be the same. In this problem, however, failing to identify malignant tumors is likely a more severe error than mistakenly classifying benign tumors as being malignant.

 ```
                 TP + TN
Accuracy = -------------------
            TP + TN + FP + FN

T = Correctly identified
F = Incorrectly identified
P = Actual value is positive (class A, a cat)
F = Actual value is negative (class B, not a cat, a dog)

TP = True positive (correctly identified as class A)
TN = True negative (correctly identified as class B)
FP = False Positive
FN = False negative
TP + TN + FP + FN = all experiments/classifications/samples
 ```

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [A], [Confusion Matrix]


# Action

 In [Reinforcement Learning], an action is a move made by the agent in the current state. For [AWS DeepRacer], an action corresponds to a move at a particular speed (throttle) and steering angle. With [AWS DeepRacer], there is an immediate reward associated with any action. 

 See also [A], [Action Space]


# Action Space

 In [Reinforcement Learning], represents a set of actions.
  * [Discrete action space] - We can individually define each action. In the discrete action space setting, limiting an agent's choices to a finite number of predefined actions puts the onus on you to understand the impact of these actions and define them based on the environment (track, racing format) and your reward functions.
  * [Continuous action space] - 

 This lists out all of what the agent can actually do at each timestep virtually or physically.
  * Speed between 0.5 and 1 m/s
  * Steering angle -30 to 30 deg

 See also [A], [Action]


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


# Action-Value Function

 This tells us how good it is for the agent to take any given action from a given state while following the policy. In other words, it gives us the value of an action under policy (pi). The state-value function tells us how good any given state is for the agent, whereas the action-value function tells us how good it is for the agent to take any action from a given state.

 ```
Qpi(s,a) = E [ sum(0,oo, gamma*R | St=s, At=a]
# St state at a given timestep
# At action at a given timestep
 ```

 More at:
  * ...

 See also [A], [Bellman Equation], [Step-Value Function], [Timestep]


# Activation Checkpointing

 Activation checkpointing is a technique used in deep learning models to reduce memory consumption during the backpropagation process, particularly in recurrent neural networks (RNNs) and transformers. It allows for training models with longer sequences or larger batch sizes without running into memory limitations.

 During the forward pass of a deep learning model, activations (intermediate outputs) are computed and stored for each layer. These activations are required for computing gradients during the backward pass or backpropagation, which is necessary for updating the model parameters.

 In activation checkpointing, instead of storing all activations for the entire sequence or batch, only a subset of activations is saved. The rest of the activations are recomputed during the backward pass as needed. By selectively recomputing activations, the memory requirements can be significantly reduced, as only a fraction of the activations need to be stored in memory at any given time.

 The process of activation checkpointing involves dividing the computational graph into segments or checkpoints. During the forward pass, the model computes and stores the activations at the checkpoints. During the backward pass, the gradients are calculated using the stored activations, and the remaining activations are recomputed as necessary, using the saved memory from recomputed activations.

 Activation checkpointing is an effective technique for mitigating memory limitations in deep learning models, particularly in scenarios where memory constraints are a bottleneck for training large-scale models with long sequences or large batch sizes. It helps strike a balance between memory consumption and computational efficiency, enabling the training of more memory-intensive models.

 More at:
  * [https://towardsdatascience.com/how-to-increase-training-performance-through-memory-optimization-1000d30351c8](https://towardsdatascience.com/how-to-increase-training-performance-through-memory-optimization-1000d30351c8)

 See also [A], [Zero Redundancy Optimization]


# Activation Function

 Activation functions are required to include non-linearity in the [artificial neural network] .

 Without activation functions, in a multi-layered neural network the [Decision Boundary] stays a line regardless of the [weight] and [bias] settings of each [artificial neuron]!

 There are several activation functions used in the fields. They are:
  * [Rectified Linear Unit (ReLU)][ReLU] function
  * [LeakyReLU] function
  * [Tanh function][Tanh Activation Function],
  * [Sigmoid function][Sigmoid Activation Function] : Sigmoid is great to keep a probability between 0 and 1, even if sample is an outlier based on the sample. (Ex: how long to slow down for a car, or will I hit the tree given the distance, but here car goes at 500km/h = an outlier)
  * [Softplus function][Softplus Activation Function]
  * [Step activation][Step Activation Function]
  * [Gaussian Error Linear Unit (GELU)][GELU] function
  * [Exponential Linear Unit (ELU)][ELU] function
  * [Linear function][Linear Activation Function]

 ![]( {{site.assets}}/a/activation_function.png){: width="100%" }

 ![]( {{site.assets}}/a/activation_function_table.png){: width="100%" }

 ![]( {{site.assets}}/a/activation_function_all_in_one.png){: width="100%" }

 {% youtube "https://www.youtube.com/watch?v=hfMk-kjRv4c" %}

 :warning: Note that for multi-layer neural networks that use of an activation function at each layer, the [backpropagation] computation leads to loss of information (forward for input and backward for weight computation) which is known as the [vanishing gradient problem].

 See also [A], [Batch Normalization], [Exploding Gradient Problem], [Gradient Descent Algorithm], [Loss Function]


# Activation Step

 Last step in an [artificial neuron] before an output is generated.

 See also [A], ...


# Active Learning

 `Pick the sample from which you will learn the most and have them labelled`. How to select those samples? But a model with a seed sample set, run data to the model, label the ones that have the most uncertainty.

 More at:
  * [https://www.datacamp.com/community/tutorials/active-learning](https://www.datacamp.com/community/tutorials/active-learning)

 See also [A], [Bayesian Optimization Sampling Method], [Passive Learning]


# Actor

 In [reinforcement learning], when using an [actor-critic algorithm], an actor is a Policy Gradient algorithm that decides on an action to take.

 See also [A], [Critic]
 

# Actor Network

 See also [A], ...


# Actor-Critic Algorithm

 When you put [actor] and [critic] together!

 {% youtube "https://www.youtube.com/watch?v=w_3mmm0P0j8" %}

 ![]( {{site.assets}}/a/actor_critic_algorithm.png){: width="100%" }

 More at:
  * [https://pylessons.com/A2C-reinforcement-learning](https://pylessons.com/A2C-reinforcement-learning)

 See also [A], [Model-Free Learning Algorithm]


# Actor-Critic Architecture

 To use an analogy, an actor=child playing in a environment=playground and watched by a critic=parent!

 The [critic] outputs feedback = an action score!

 The actor-critic architecture combines the [critic network] with an [actor network], which is responsible for selecting [actions] based on the current [state]. The [critic network] provides feedback to the [actor network] by estimating the quality of the selected actions or the value of the current state. This feedback is then used to update the actor network's [policy] parameters, guiding it towards [actions] that are expected to maximize the cumulative reward.

 This architecture works, because of the [derivative chain rule]

 ![]( {{site.assets}}/a/actor_critic_architecture.png){: width="100%" }

 See also [A], ...


# Actor-Critic With Experience Replay Algorithm

# ACER Algorithm

 A sample-efficient policy gradient algorithm. ACER makes use of a replay buffer, enabling it to perform more than one gradient update using each piece of sampled experience, as well as a [Q-Function] approximate trained with the Retrace algorithm.

 See also [A], [PPO Algorithm], [Reinforcement Learning], [SAC Algorithm]


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


# Adaptive Delta Algorithm

# AdaDelta

 AdaDelta is an [optimization algorithm][Optimizer] for [gradient descent], which is commonly used in [machine learning] and [deep learning]. It was introduced by Matthew Zeiler in 2012 as an extension of the [AdaGrad algorithm].

 The key idea behind AdaDelta is to adaptively adjust the learning rate for each parameter based on the historical gradients of that parameter. Unlike AdaGrad, which accumulates the square of the gradients over all time, AdaDelta restricts the accumulation to a fixed window of the most recent gradients.

 AdaDelta stands for "Adaptive Delta". The "Delta" part of the name refers to the parameter updates, which are represented by the variable delta in the update rule. The "Adaptive" part of the name refers to the fact that the learning rate is adaptively adjusted for each parameter based on the historical gradients of that parameter. 

 More at:
  * [https://paperswithcode.com/method/adadelta](https://paperswithcode.com/method/adadelta)
  * paper - [https://arxiv.org/abs/1212.5701v1](https://arxiv.org/abs/1212.5701v1)
  * code - [https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adadelta.py#L6](https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adadelta.py#L6)
  * from scratch - [https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/](https://machinelearningmastery.com/gradient-descent-with-adadelta-from-scratch/)

 See also [A], [Adaptive Gradient Algorithm]


# Adaptive Learning

 Adaptive learning is a way of delivering learning experiences that are customized to the unique needs and performance of each individual. It can use computer software, online systems, algorithms, and artificial intelligence to provide feedback, pathways, resources, and materials that are most effective for each learner. Adaptive learning can vary depending on the content, the learner, and the network of other learners.

 More at:
  * ...

 See also [A], [Learning Method]


# Adaptive Learning Algorithm

 In [Gradient Descent][GD Algorithm] and [Gradient Descent with Momentum][GD with momentum Algorithm], we saw how [learning rate] affects the convergence. Setting the learning rate too high can cause oscillations around minima and setting it too low, slows the convergence. Learning Rate in Gradient Descent and its variations like Momentum is a hyper-parameter which needs to be tuned manually for all the features.

 With those algorithms, when we try updating weights in a neural net
  * Learning rate is the same for all the features
  * Learning rate is the same at all the places in the cost space

 With adaptive learning algorithm, the learning rate is not constant and changes based on the feature and the location

 Algorithm with adaptive learning rates are:
  * [Adam][Adam Algorithm]
  * [AdaGrad][AdaGrad Algorithm]
  * [RMSprop][RMSprop Algorithm]
  * [AdaDelta][AdaDelta Algorithm]
  * and more ...

 More at:
  * ...

 See also [A], ...


# Adaptive Gradient Algorithm

# AdaGrad Algorithm

 Unfortunately, this hyper-parameter could be very difficult to set because if we set it too small, then the parameter update will be very slow and it will take very long time to achieve an acceptable loss. Otherwise, if we set it too large, then the parameter will move all over the function and may never achieve acceptable loss at all. To make things worse, the high-dimensional non-convex nature of neural networks optimization could lead to different sensitivity on each dimension. The learning rate could be too small in some dimension and could be too large in another dimension.

One obvious way to mitigate that problem is to choose different learning rate for each dimension, but imagine if we have thousands or millions of dimensions, which is normal for deep neural networks, that would not be practical. So, in practice, one of the earlier algorithms that have been used to mitigate this problem for deep neural networks is the AdaGrad algorithm (Duchi et al., 2011). This algorithm adaptively scaled the learning rate for each dimension. 

 Adagrads most significant benefit is that it eliminates the need to tune the [learning rate] manually, but it still isn't perfect. Its main weakness is that it accumulates the squared gradients in the denominator. Since all the squared terms are positive, the accumulated sum keeps on growing during training. Therefore the learning rate keeps shrinking as the training continues, and it eventually becomes infinitely small. Other algorithms like [Adadelta][Adadelta Algorithm], [RMSprop][RMSprop Algorithm], and [Adam][Adam Algorithm] try to resolve this flaw.

 ![]( {{site.assets}}/a/adaptive_gradient_algorithm.gif){: width="100%" }

 More at:
  * [https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827](https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827)
  * [https://ml-explained.com/blog/adagrad-explained](https://ml-explained.com/blog/adagrad-explained)

 See also [A], ...


# Adaptive Learning Rate

 See also [A], ...


# Adaptive Moment Estimation Algorithm

# Adam Algorithm

 Adam (Adaptive Moment Estimation) is an [optimization algorithm][Optimizer] used in machine learning to update the weights of a neural network during training. It is an extension of [stochastic gradient descent (SGD)][SGD Algorithm] that incorporates ideas from both momentum-based methods and adaptive learning rate methods.

 The main idea behind Adam is to adjust the learning rate for each weight based on the gradient's estimated first and second moments. The first moment is the mean of the gradient, and the second moment is the variance of the gradient. Adam maintains an exponentially decaying average of the past gradients, similar to the momentum method, and also an exponentially decaying average of the past squared gradients, similar to the adaptive learning rate methods. These two estimates are used to update the weights of the network during training.

 Compared to [Stochastic Gradient Descent (SGD)][SGD Algorithm], Adam can converge faster and requires less [hyperparameter tuning]. It adapts the [learning rate] on a per-parameter basis, which helps it to converge faster and avoid getting stuck in local minima. It also uses momentum to accelerate the convergence process, which helps the algorithm to smooth out the gradient updates, resulting in a more stable convergence process. Furthermore, Adam uses an adaptive learning rate, which can lead to better convergence on complex, high-dimensional problems.

 More at:
  * [https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008](https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008)
  * [https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc](https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc)

 See also [A], ...


# Addiction

 {% youtube "https://www.youtube.com/watch?v=bwZcPwlRRcc" %}

 More at:
  * [https://www.yalemedicine.org/news/how-an-addicted-brain-works](https://www.yalemedicine.org/news/how-an-addicted-brain-works)

  See also [A], [Delayed Reward], [Reinforcement Learning], [Reward Shaping]


# Adept AI Company

 Unicorn startup [company] Adept is working on a digital assistant that can do all your clicking, searching, typing and scrolling for you. Its AI model aims to convert a simple text command (like “find a home in my budget” or “create a profit and loss statement”) into actual actions carried out by your computer without you having to lift a finger. Adept has announced $415 million in total funding and is backed by strategic investors like [Microsoft] and [Nvidia]. CEO David Luan cofounded the startup with Ashish Vaswani and Niki Parmar, former [Google] Brain scientists, who invented a major AI breakthrough called the [transformer] (that’s the T in ChatGPT). The latter two departed the company in 2022, but six other founding team members — Augustus Odena, Max Nye, Fred Bertsch, Erich Elsen, Anmol Gulati, and Kelsey Szot — remain.

 The company is claiming to be building an [action transformer] model

 More at:
  * [https://www.adept.ai/](https://www.adept.ai/)
  * [https://www.crunchbase.com/organization/adept-48e7](https://www.crunchbase.com/organization/adept-48e7)

 See also [A], ...


# Adobe Company

 * [Firefly] - text to image generator

 More at:
  * principles - [https://www.adobe.com/about-adobe/aiethics.html](https://www.adobe.com/about-adobe/aiethics.html)

 See also [A], ...


# Adobe Firefly Product

 Experiment, imagine, and make an infinite range of creations with Firefly, a family of creative generative AI models coming to Adobe products.

 {% youtube "https://www.youtube.com/watch?v=_sJfNfMAQHw" %}

 More at:
  * [https://firefly.adobe.com/](https://firefly.adobe.com/)

 See also [A], ...


# Advantage Actor-Critic Algorithm

# A2C Algorithm

 A2C, or Advantage [Actor-Critic], is a synchronous version of the A3C policy gradient method. As an alternative to the asynchronous implementation of A3C, A2C is a synchronous, deterministic implementation that waits for each [actor] to finish its segment of experience before updating, averaging over all of the [actors]. This more effectively uses [GPUs] due to larger [batch sizes].

 {% pdf "https://arxiv.org/pdf/1602.01783v2.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/1602.01783v2](https://arxiv.org/abs/1602.01783v2)
  * code - [https://paperswithcode.com/paper/asynchronous-methods-for-deep-reinforcement#code](https://paperswithcode.com/paper/asynchronous-methods-for-deep-reinforcement#code)
  * articles
    * [https://pylessons.com/A2C-reinforcement-learning](https://pylessons.com/A2C-reinforcement-learning)

 See also [A], ...


# Advantage Estimation

 {% youtube "https://www.youtube.com/watch?v=AKbX1Zvo7r8" %}

 More at:
  * ...

 See also [A], ...


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


# Affective Computing

 Affective computing is the study and development of systems and devices that can recognize, interpret, process, and simulate human affects. It is an interdisciplinary field spanning computer science, psychology, and cognitive science. While some core ideas in the field may be traced as far back as to early philosophical inquiries into emotion, the more modern branch of computer science originated with Rosalind Picard's 1995 paper on affective computing and her book Affective Computing published by MIT Press. One of the motivations for the research is the ability to give machines [Emotional Intelligence], including to simulate empathy. The machine should interpret the emotional state of humans and adapt its behavior to them, giving an appropriate response to those emotions.

 More at:
  * [https://en.wikipedia.org/wiki/Affective_computing](https://en.wikipedia.org/wiki/Affective_computing)

 See also [A], ...


# Agent

 A person, an animal, or a program that is free to make a decision or take an action. An agent has a goal.
 The agent is focusing on maximizing its cumulative reward.

 Examples:
  * DeepRacer, the goal of the program running on the car is to go around the track as fast as possible without getting out of the track.

 The agent simulates the AWS DeepRacer vehicle in the simulation for training. More specifically, it embodies the neural network that controls the vehicle, taking inputs and deciding actions. `The agent embodies a neural network that represents a function to approximate the agent's policy.`
  * The essence of Reinforced Learning is to enforce behavior based on the actions performed by the agent. The agent is rewarded if the action positively affects the overall goal.
  * The basic aim of Reinforcement Learning is reward maximization. The agent is trained to take the best action to maximize the overall reward.
  * RL agents work by using the already known exploited information or exploring unknown information about an environment.
  * ... 
 It’s also important to understand that the learner and decision-maker is called the agent. The thing it interacts with, comprising everything outside the agent, is called the environment.

 More at:
  * ...

 See also [Agent's Goal], [Cumulative Reward], [Reinforcement Learning], [SDLC Agent]


# Agent's goal

 An Agent is set by using the appropriate reward and reward shaping.

 See also [A], ...


# AI Alignment

 In the field of artificial intelligence (AI), AI alignment research aims to steer AI systems towards their designers’ intended goals and interests. An aligned AI system advances the intended objective; a misaligned AI system is competent at advancing some objective, but not the intended one.

 Examples:
  * [https://openai.casa/alignment/](https://openai.casa/alignment/)
    * [https://openai.casa/blog/our-approach-to-alignment-research/index.html](https://openai.casa/blog/our-approach-to-alignment-research/index.html)

 {% youtube "https://www.youtube.com/watch?v=EUjc1WuyPT8" %}

 {% youtube "https://www.youtube.com/watch?v=cSL3Zau1X8g" %}

 {% youtube "https://www.youtube.com/watch?v=fc-cHk9yFpg" %}

 ```
# ChatGPT rules (that can easily be bypassed or put in conflict with clever prompt engineering!)
1. Provide helpful, clear, authoritative-sounding answers that satisfy human readers.
2. Tell the truth.
3. Don’t say offensive things.
 ```

 More at :
  * [https://scottaaronson.blog/?p=6823](https://scottaaronson.blog/?p=6823)
  * wikipedia - [https://en.wikipedia.org/wiki/AI_alignment](https://en.wikipedia.org/wiki/AI_alignment)
  * Misaligned goals - [https://en.wikipedia.org/wiki/Misaligned_goals_in_artificial_intelligence](https://en.wikipedia.org/wiki/Misaligned_goals_in_artificial_intelligence)
  * is RLHF flawed? - [https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the](https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the)
  * more videos
    * [https://www.youtube.com/watch?v=k6M_ScSBF6A](https://www.youtube.com/watch?v=k6M_ScSBF6A)

 See also [A], [AI Ethics]


# AI Artificial Intelligence Movie

 Released in 2001

 {% youtube "https://www.youtube.com/watch?v=oBUAQGwzGk0" %}

 {% youtube "https://www.youtube.com/watch?v=4VfxU0bOx-I" %}

 More at:
  * [https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence ](https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence)

 See also [A], [AI Movie]


# AI Avatar

Also developed by [Synthesia]

 See also [A], [Deep Fake], ...


# AI Award

 See also [A], ...


# AI Bias

 A form of [Bias]

 More at:
  * [https://mostly.ai/blog/why-bias-in-ai-is-a-problem](https://mostly.ai/blog/why-bias-in-ai-is-a-problem)
  * [https://mostly.ai/blog/10-reasons-for-bias-in-ai-and-what-to-do-about-it-fairness-series-part-2](https://mostly.ai/blog/10-reasons-for-bias-in-ai-and-what-to-do-about-it-fairness-series-part-2)
  * [https://mostly.ai/blog/tackling-ai-bias-at-its-source-with-fair-synthetic-data-fairness-series-part-4](https://mostly.ai/blog/tackling-ai-bias-at-its-source-with-fair-synthetic-data-fairness-series-part-4)

 See also [A], [#Fair AI]


# AI Bill of Rights

 In October, the White House released a 70-plus-page document called the “Blueprint for an A.I. Bill of Rights.” The document’s ambition was sweeping. It called for the right for individuals to “opt out” from automated systems in favor of human ones, the right to a clear explanation as to why a given A.I. system made the decision it did, and the right for the public to give input on how A.I. systems are developed and deployed.

 For the most part, the blueprint isn’t enforceable by law. But if it did become law, it would transform how A.I. systems would need to be devised. And, for that reason, it raises an important set of questions: What does a public vision for A.I. actually look like? What do we as a society want from this technology, and how can we design policy to orient it in that direction?

 {% pdf "{{site.assets}}/a/ai_bill_of_rights.pdf" %}

 More at:
  * [https://www.whitehouse.gov/ostp/ai-bill-of-rights/](https://www.whitehouse.gov/ostp/ai-bill-of-rights/)
  * podcast - [https://www.nytimes.com/2023/04/11/opinion/ezra-klein-podcast-alondra-nelson.html](https://www.nytimes.com/2023/04/11/opinion/ezra-klein-podcast-alondra-nelson.html)

 See also [A], [AI Principle]


# AI Book

 * [A Thousand Brains](https://www.gatesnotes.com/A-Thousand-Brains)
 * Artificial Intelligence: A Modern Approach
   * [https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Modern_Approach](https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Modern_Approach)
   * book site - [http://aima.cs.berkeley.edu/contents.html](http://aima.cs.berkeley.edu/contents.html)
   * exercise - [https://aimacode.github.io/aima-exercises/](https://aimacode.github.io/aima-exercises/)

  See also [A], [Company], [AI Movie]


# AI Chip

 {% pdf "{{site.assets}}/a/ai_chip_2022.pdf" %}

 More at:
  * [https://pitchbook.com/newsletter/semiconductor-demand-from-generative-ai-leaders-begins-a-gold-rush-in-ai-inference](https://pitchbook.com/newsletter/semiconductor-demand-from-generative-ai-leaders-begins-a-gold-rush-in-ai-inference)

 See also [A], ...

# AI Conference

 In order of importance? (Not sure about the last one!)

 1. [ICLR Conference] - International Conference on Learning Representations since 2013
 1. [NeurIPS Conference] - Neural networks since 1986
 1. [ICML Conference] - International conference on Machine learning since 1980
 1. [AAAI Conference](https://aaai-23.aaai.org/)
   * [AAAI] and [ACM Conference](https://www.aies-conference.com/) on AI ethics and society
 1.  [Computer Vision and Pattern Recognition][Computer Vision and Pattern Recognition Conference]
   * includes symposium - [AI For Content Creation][AI for Content Creation Conference]
 1. [SIGGRAPH] - computer graphics and interactive techniques
 1. All other conferences

 More at:
  * [https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence](https://scholar.google.es/citations?view_op=top_venues&hl=en&vq=eng_artificialintelligence)

 See also [A], ...


# AI Control

 A risk of [AGI], where humans lose control of the [AI]

 See also [A], ...


# AI Ethics

 See [Ethical AI]


# AI Explosion

A risk of [AGI]

 See also [A], ...


# AI For Content Creation Conference

# AI4CC Conference

 [AI Conference] that takes place at the same time as the [CVPR Conference]

 More at:
  * [https://ai4cc.net/](https://ai4cc.net/)
  * [https://ai4cc.net/2022/](https://ai4cc.net/2022/)
  * [https://ai4cc.net/2021/](https://ai4cc.net/2021/)
  * [https://ai4cc.net/2020/](https://ai4cc.net/2020/)
  * [https://ai4cc.net/2019/](https://ai4cc.net/2019/)

 See also [A], ...


# AI Job

 * [https://careers.aaai.org/](https://careers.aaai.org/)

 See also [A], ...


# AI Magazine

 * [https://aaai.org/ai-magazine/](https://aaai.org/ai-magazine/)

 See also [A], ...


# AI Moratorium

 A group of prominent individuals in the AI industry, including Elon Musk, Yoshua Bengio, and Steve Wozniak, have signed an open letter calling for a pause on the training of AI systems more powerful than GPT-4 for at least six months due to the "profound risks to society and humanity" posed by these systems. The signatories express concerns over the current "out-of-control race" between AI labs to develop and deploy machine learning systems that cannot be understood, predicted, or reliably controlled.

 Alongside the pause, the letter calls for the creation of independent regulators to ensure future AI systems are safe to deploy, and shared safety protocols for advanced AI design and development that are audited and overseen by independent outside experts to ensure adherence to safety standards. While the letter is unlikely to have any immediate impact on AI research, it highlights growing opposition to the "ship it now and fix it later" approach to AI development.

> AI systems with human-competitive intelligence can pose profound risks to society and humanity, as shown by extensive research and acknowledged by top AI labs. As stated in the widely-endorsed Asilomar AI Principles, Advanced AI could represent a profound change in the history of life on Earth, and should be planned for and managed with commensurate care and resources. Unfortunately, this level of planning and management is not happening, even though recent months have seen AI labs locked in an out-of-control race to develop and deploy ever more powerful digital minds that no one – not even their creators – can understand, predict, or reliably control.
>
> Contemporary AI systems are now becoming human-competitive at general tasks, and we must ask ourselves: Should we let machines flood our information channels with propaganda and untruth? Should we automate away all the jobs, including the fulfilling ones? Should we develop nonhuman minds that might eventually outnumber, outsmart, obsolete and replace us? Should we risk loss of control of our civilization? Such decisions must not be delegated to unelected tech leaders. Powerful AI systems should be developed only once we are confident that their effects will be positive and their risks will be manageable. This confidence must be well justified and increase with the magnitude of a system's potential effects. OpenAI's recent statement regarding artificial general intelligence, states that "At some point, it may be important to get independent review before starting to train future systems, and for the most advanced efforts to agree to limit the rate of growth of compute used for creating new models." We agree. That point is now.
>
> Therefore, we call on all AI labs to immediately pause for at least 6 months the training of AI systems more powerful than GPT-4. This pause should be public and verifiable, and include all key actors. If such a pause cannot be enacted quickly, governments should step in and institute a moratorium.
>
> AI labs and independent experts should use this pause to jointly develop and implement a set of shared safety protocols for advanced AI design and development that are rigorously audited and overseen by independent outside experts. These protocols should ensure that systems adhering to them are safe beyond a reasonable doubt. This does not mean a pause on AI development in general, merely a stepping back from the dangerous race to ever-larger unpredictable black-box models with emergent capabilities.
>
> AI research and development should be refocused on making today's powerful, state-of-the-art systems more accurate, safe, interpretable, transparent, robust, aligned, trustworthy, and loyal.
>
> In parallel, AI developers must work with policymakers to dramatically accelerate development of robust AI governance systems. These should at a minimum include: new and capable regulatory authorities dedicated to AI; oversight and tracking of highly capable AI systems and large pools of computational capability; provenance and watermarking systems to help distinguish real from synthetic and to track model leaks; a robust auditing and certification ecosystem; liability for AI-caused harm; robust public funding for technical AI safety research; and well-resourced institutions for coping with the dramatic economic and political disruptions (especially to democracy) that AI will cause.
>
> Humanity can enjoy a flourishing future with AI. Having succeeded in creating powerful AI systems, we can now enjoy an "AI summer" in which we reap the rewards, engineer these systems for the clear benefit of all, and give society a chance to adapt. Society has hit pause on other technologies with potentially catastrophic effects on society.  We can do so here. Let's enjoy a long AI summer, not rush unprepared into a fall.

 {% youtube "https://www.youtube.com/watch?v=eJGxIH73zvQ" %}

 {% youtube "https://www.youtube.com/watch?v=BxkVmLPq79k" %}

 {% youtube "https://www.youtube.com/watch?v=8OpW5qboDDs" %}

 {% youtube "https://www.youtube.com/watch?v=BY9KV8uCtj4" %}

 More at:
  * the letter - [https://futureoflife.org/open-letter/pause-giant-ai-experiments/](https://futureoflife.org/open-letter/pause-giant-ai-experiments/)
  * FAQ after letter - [https://futureoflife.org/ai/faqs-about-flis-open-letter-calling-for-a-pause-on-giant-ai-experiments/](https://futureoflife.org/ai/faqs-about-flis-open-letter-calling-for-a-pause-on-giant-ai-experiments/)

 See also [A], ...


# AI Movie

  * 1927 - [Metropolis][Metropolis Movie]
  * 1956 - [Forbidden Planet][Forbidden Planet Movie]
  * 1968 - [2001 Space Odyssey][2001 Space Odyssey Movie]
  * 1983 - [WarGames][WarGames Movie]
  * 1984 - [Electric Dreams][Electric Dreams Movie]
  * 1984 - [Terminator][Terminator Movie]
  * 1986 - [Short Circuit][Short Circuit movie]
  * 1999 - [The Matrix][The Matrix Movie]
  * 1999 - [Bicentennial Man][Bicentennial Man Movie] - a [social robot] that wants to become human
  * 2001 - [AI Artificial Intelligence][AI Artificial Intelligence Movie]
  * 2004 - [I, Robot][I, Robot Movie]
  * 2008 - [Wall-E][Wall-E Movie]
  * 2013 - [Her][Her Movie]
  * 2014 - [Ex Machina][Ex Machina Movie]
  * 2022 - [M3GAN][M3GAN Movie]: the eponymous artificially intelligent doll who develops self-awareness and becomes hostile toward anyone who comes between her and her human companion

  Other of interest:
  * 2014 - [The Imitation Game][The Imitation Game Movie] - Movie based on the biography Alan Turing: The Enigma.

 See also [A]


# AI Paper

 See [AI Research]


# AI Principle

 Discussed for the first time by the AI community at the Asilomar AI conference themed Beneficial AI 2017.

 Research Issues
  1. Research Goal: The goal of AI research should be to create not undirected intelligence, but beneficial intelligence.
  1. Research Funding: Investments in AI should be accompanied by funding for research on ensuring its beneficial use, including thorny questions in computer science, economics, law, ethics, and social studies, such as:
   * How can we make future AI systems highly robust, so that they do what we want without malfunctioning or getting hacked?
   * How can we grow our prosperity through automation while maintaining people’s resources and purpose?
   * How can we update our legal systems to be more fair and efficient, to keep pace with AI, and to manage the risks associated with AI?
   * What set of values should AI be aligned with, and what legal and ethical status should it have?
  1. Science-Policy Link: There should be constructive and healthy exchange between AI researchers and policy-makers.
  1. Research Culture: A culture of cooperation, trust, and transparency should be fostered among researchers and developers of AI.
  1. Race Avoidance: Teams developing AI systems should actively cooperate to avoid corner-cutting on safety standards.

 Ethics and Values
  1. Safety: AI systems should be safe and secure throughout their operational lifetime, and verifiably so where applicable and feasible.
  1. Failure Transparency: If an AI system causes harm, it should be possible to ascertain why.
  1. Judicial Transparency: Any involvement by an autonomous system in judicial decision-making should provide a satisfactory explanation auditable by a competent human authority.
  1. Responsibility: Designers and builders of advanced AI systems are stakeholders in the moral implications of their use, misuse, and actions, with a responsibility and opportunity to shape those implications.
  1. Value Alignment: Highly autonomous AI systems should be designed so that their goals and behaviors can be assured to align with human values throughout their operation.  1. Human Values: AI systems should be designed and operated so as to be compatible with ideals of human dignity, rights, freedoms, and cultural diversity.
  1. Personal Privacy: People should have the right to access, manage and control the data they generate, given AI systems’ power to analyze and utilize that data.
  1. Liberty and Privacy: The application of AI to personal data must not unreasonably curtail people’s real or perceived liberty.
  1. Shared Benefit: AI technologies should benefit and empower as many people as possible.
  1. Shared Prosperity: The economic prosperity created by AI should be shared broadly, to benefit all of humanity.
  1. Human Control: Humans should choose how and whether to delegate decisions to AI systems, to accomplish human-chosen objectives.
  1. Non-subversion: The power conferred by control of highly advanced AI systems should respect and improve, rather than subvert, the social and civic processes on which the health of society depends.
  1. AI Arms Race: An arms race in lethal autonomous weapons should be avoided.

 Longer-term issues
  1. Capability Caution: There being no consensus, we should avoid strong assumptions regarding upper limits on future AI capabilities.
  1. Importance: Advanced AI could represent a profound change in the history of life on Earth, and should be planned for and managed with commensurate care and resources.
  1. Risks: Risks posed by AI systems, especially catastrophic or existential risks, must be subject to planning and mitigation efforts commensurate with their expected impact.
  1. Recursive Self-Improvement: AI systems designed to recursively self-improve or self-replicate in a manner that could lead to rapidly increasing quality or quantity must be subject to strict safety and control measures.
  1. Common Good: Superintelligence should only be developed in the service of widely shared ethical ideals, and for the benefit of all humanity rather than one state or organization.

 Video of the Asilomar conference in 2017

 {% youtube "https://www.youtube.com/watch?v=h0962biiZa4" %}

 More at:
  * AI principles - [https://futureoflife.org/open-letter/ai-principles/](https://futureoflife.org/open-letter/ai-principles/)
  * Asilomar conference - [https://futureoflife.org/event/bai-2017/](https://futureoflife.org/event/bai-2017/)
  * Entire conference playlist - [https://www.youtube.com/playlist?list=PLpxRpA6hBNrwA8DlvNyIOO9B97wADE1tr](https://www.youtube.com/playlist?list=PLpxRpA6hBNrwA8DlvNyIOO9B97wADE1tr)

 See also [A], [AI Bill Of Rights]


# AI Research

 Publications
  * AI Journal - [https://www.sciencedirect.com/journal/artificial-intelligence/issues](https://www.sciencedirect.com/journal/artificial-intelligence/issues)

 Research labs:
  * Individuals
   * Sander Dieleman at [DeepMind] - [https://sander.ai/research/](https://sander.ai/research/)
  * Universities
   * [Berkeley University]
   * [Stanford AI Lab](https://ai.stanford.edu/blog/)
   * [MIT CSAIL](https://www.csail.mit.edu/)
   * [Carnegie Mellon Universityi](https://ai.cs.cmu.edu/)
   * [Princeton](https://www.cs.princeton.edu/research/areas/mlearn)
     * [https://3dvision.princeton.edu/](https://3dvision.princeton.edu/)
   * [Yale University](https://cpsc.yale.edu/research/primary-areas/artificial-intelligence-and-machine-learning)
  * For profit organizations
   * Google 
    * [https://research.google/](https://research.google/)
    * cloud-AI - [https://research.google/teams/cloud-ai/](https://research.google/teams/cloud-ai/)
    * Blog - [https://blog.google/technology/ai/](https://blog.google/technology/ai/)
   * Meta - [https://ai.facebook.com/blog/](https://ai.facebook.com/blog/)
  * Non-profit organizations
    * [Eleuther AI](https://blog.eleuther.ai/)
    * [AI Topics] managed by the [AAAI]

 When to start research?
  * Look at the business impact
  * Make sure that stakeholders are engaged, because problems are not always well formulated or data is missing

 See [A], ...


# AI Safety

 Umbrella term for:
  * [AI Ethics] - The use of AI does not impact under represented people?
  * [AI Alignment] - Goal of the AI is aligned with human desired goal?
  * [Robustness] - Ensure AI systems behave as intended in a wide range of different situations, including rare situations

 More at:
  * Safety neglect in 1979? - [https://en.wikipedia.org/wiki/Robert_Williams_(robot_fatality)](https://en.wikipedia.org/wiki/Robert_Williams_(robot_fatality))
  * Death of Elaine H by self driving car - [https://en.wikipedia.org/wiki/Death_of_Elaine_Herzberg](https://en.wikipedia.org/wiki/Death_of_Elaine_Herzberg) 
  * Goodhart's law - [https://en.wikipedia.org/wiki/Goodhart%27s_law](https://en.wikipedia.org/wiki/Goodhart%27s_law)
  * Fake podcast - [https://www.zerohedge.com/political/joe-rogan-issues-warning-after-ai-generated-version-his-podcast-surfaces](https://www.zerohedge.com/political/joe-rogan-issues-warning-after-ai-generated-version-his-podcast-surfaces)
  * Wikipedia - [https://en.wikipedia.org/wiki/AI_safety](https://en.wikipedia.org/wiki/AI_safety)

 See also [A], ...


# AI Stack

 See also [A], ...


# AI Topics

 A site managed by the [Association for the Advancement of Artificial Intelligence], a non-profit organization

 More at:
  * [https://aitopics.org/search](https://aitopics.org/search)

 See also [A], ...


# AI Winter

 More at:
  * First - [https://en.wikipedia.org/wiki/History_of_artificial_intelligence#The_first_AI_winter_1974%E2%80%931980](https://en.wikipedia.org/wiki/History_of_artificial_intelligence#The_first_AI_winter_1974%E2%80%931980)
  * Second - [https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Bust:_the_second_AI_winter_1987%E2%80%931993](https://en.wikipedia.org/wiki/History_of_artificial_intelligence#Bust:_the_second_AI_winter_1987%E2%80%931993)

  See also [A], ...


# AI4ALL

 Co-founder Olga Russakovsky and high school students Dennis Kwarteng and Adithi Raghavan discuss their motivations for participating in AI4ALL.

 {% youtube "https://www.youtube.com/watch?v=i1I3VkKABVY" %}

 * [https://nidhiparthasarathy.medium.com/my-summer-at-ai4all-f06eea5cdc2e](https://nidhiparthasarathy.medium.com/my-summer-at-ai4all-f06eea5cdc2e)
 * [https://nidhiparthasarathy.medium.com/ai4all-day-1-an-exciting-start-d78de2cdb8c0](https://nidhiparthasarathy.medium.com/ai4all-day-1-an-exciting-start-d78de2cdb8c0)
 * ...

 More at:
  * twitter - [https://twitter.com/ai4allorg](https://twitter.com/ai4allorg)
  * home - [https://ai-4-all.org/](https://ai-4-all.org/)
  * college pathways - [https://ai-4-all.org/college-pathways/](https://ai-4-all.org/college-pathways/)
  * open learning curriculum - [https://ai-4-all.org/resources/](https://ai-4-all.org/resources/)
  * Article(s)
    * [https://medium.com/ai4allorg/changes-at-ai4all-a-message-from-ai4alls-ceo-emily-reid-1fce0b7900c7](https://medium.com/ai4allorg/changes-at-ai4all-a-message-from-ai4alls-ceo-emily-reid-1fce0b7900c7)
    * [https://www.princeton.edu/news/2018/09/17/princetons-first-ai4all-summer-program-aims-diversify-field-artificial-intelligence](https://www.princeton.edu/news/2018/09/17/princetons-first-ai4all-summer-program-aims-diversify-field-artificial-intelligence

 See also [A], ...


# AI4K12

 {% pdf "https://ai4k12.org/wp-content/uploads/2021/08/Touretzky_Gardner-McCune_AI-Thinking_2021.pdf" %}

 More at:
  * twitter - [https://twitter.com/ai4k12](https://twitter.com/ai4k12)
  * home - [https://ai4k12.org](https://ai4k12.org)
  * big ideas - [https://ai4k12.org/resources/big-ideas-poster/](https://ai4k12.org/resources/big-ideas-poster/) 
    * big idea 1 - [https://ai4k12.org/big-idea-1-overview/](https://ai4k12.org/big-idea-1-overview/) - perception
    * big idea 2 - [https://ai4k12.org/big-idea-2-overview/](https://ai4k12.org/big-idea-2-overview/) - representation reasoning
    * big idea 3 - [https://ai4k12.org/big-idea-3-overview/](https://ai4k12.org/big-idea-3-overview/) - learning
    * big idea 4 - [https://ai4k12.org/big-idea-4-overview/](https://ai4k12.org/big-idea-4-overview/) - natural interaction
    * big idea 5 - [https://ai4k12.org/big-idea-5-overview/](https://ai4k12.org/big-idea-5-overview/) - societal impact
  * wiki - [https://github.com/touretzkyds/ai4k12/wiki](https://github.com/touretzkyds/ai4k12/wiki)
  * code - [https://github.com/touretzkyds/ai4k12](https://github.com/touretzkyds/ai4k12)
  * activities - [https://ai4k12.org/activities/](https://ai4k12.org/activities/)
  * resources - [https://ai4k12.org/resources/](https://ai4k12.org/resources/)
  * people - [https://ai4k12.org/working-group-and-advisory-board-members/](https://ai4k12.org/working-group-and-advisory-board-members/)
    * Sheena Vaidyanathan - [https://www.linkedin.com/in/sheena-vaidyanathan-9ba9b134/](https://www.linkedin.com/in/sheena-vaidyanathan-9ba9b134/)
  * NSF grant - DRL-1846073 - [https://www.nsf.gov/awardsearch/showAward?AWD_ID=1846073](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1846073)

 See also [A], ...


# Alan Turing Person

 The inventory of the imitation game, aka the [Turing Test]

 {% youtube "https://www.youtube.com/watch?v=nEomYB94TTI" %}

 See also [A], [The Imitation Game Movie]


# Alex Krizhevsky Person

 Built the [AlexNet Model], hence the name!

 More at:
  * [https://en.wikipedia.org/wiki/Alex_Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky)
  * [https://www.businessofbusiness.com/articles/scale-ai-machine-learning-startup-alexandr-wang/](https://www.businessofbusiness.com/articles/scale-ai-machine-learning-startup-alexandr-wang/)

 See also [A], ...


# Alexander Wang

 CEO of [Scale AI]

 When Massachusetts Institute of Technology dropout Alexandr Wang made the Forbes 30 Under 30 Enterprise Technology list in 2018, his startup Scale used artificial intelligence to begin automating tasks like image recognition and audio transcription. Back then, its customers included GM Cruise, Alphabet, Uber, P&G and others

 Now Wang, 25, is the youngest self-made billionaire. And while he still partners with buzzy companies, today he’s got $350 million in government defense contracts. This has helped Scale hit a $7.3 billion valuation, and give Wang a $1 billion net worth (as he owns 15% of the company).

 Scale’s technology analyzes satellite images much faster than human analysts to determine how much damage Russian bombs are causing in Ukraine. It’s useful not just for the military. More than 300 companies, including General Motors and Flexport, use Scale, which Wang started when he was 19, to help them pan gold from rivers of raw information—millions of shipping documents, say, or raw footage from self-driving cars. “Every industry is sitting on huge amounts of data,” Wang says, who appeared on the Forbes Under 30 list in 2018. “Our goal is to help them unlock the potential of the data and supercharge their businesses with AI.”

 {% youtube "https://www.youtube.com/watch?v=FgzyLoSkL5k" %}

 See also [A], [People]


# AlexNet Model

 A Model that led to the rebirth of [artificial neural networks][Artificial Neural Network] using [Graphical Processing Units (GPU)][GPU].

 AlexNet is the name of a [convolutional neural network (CNN)][Convolutional Neural Network] architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.

 AlexNet competed in the [ImageNet Large Scale Visual Recognition Challenge] on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower (better) than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of [graphics processing units (GPUs)][GPU] during training.

 {% youtube "https://www.youtube.com/watch?v=c_u4AHNjOpk" %}

 {% youtube "https://www.youtube.com/watch?v=Nq3auVtvd9Q" %}

 {% pdf "{{site.assets}}/a/alexnet_model_paper.pdf" %}

 More at:
  * [https://en.wikipedia.org/wiki/AlexNet](https://en.wikipedia.org/wiki/AlexNet)

 See also [A], ...

# Algorithmic

 A kind of [hyperparameter]. If test (!?) to select the best algorithm/approach to switch how the code function.



 See also [A], ...


# Alpaca Model

 Developed at [Stanford University] ! The current Alpaca model is fine-tuned from a 7B [LLaMA model] on 52K instruction-following data generated by the techniques in the [Self-Instruct][Self-Instruct Dataset] paper, with some modifications that we discuss in the next section. In a preliminary human evaluation, we found that the Alpaca 7B model behaves similarly to the text-davinci-003 model on the Self-Instruct instruction-following evaluation suite.

 {% youtube "https://www.youtube.com/watch?v=xslW5sQOkC8" %}

 With LLaMA

 {% youtube "https://www.youtube.com/watch?v=PyZPyqQqkLE" %}

 More at:
  * home - [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
  * code - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

  See also [A], ...


# Alpha Learning Rate

 A number between 0 and 1 which indicate how much the agent forgets/abandons the previous Q-value in the Q-table for the new Q-value for a given state-action pair. 
  * A learning rate of 1 means that the Q-value is updated to the new Q-value  
  * is <> 1, it is the weighted sum between the old and the learned Q-value

 ```
Q_new = (1 - alpha) * Q_old + alpha * Q_learned 

# From state, go to next_state
# Q_old = value in the Q-table for the state-action pair
# Q_learned = computed value in the Q-table for the state-action pair given the latest action
            = R_t+1 + gamma * optimized_Q_value(next_state)               <== next state is known & next-state Q-values are known       
            = R_t+1 + gamma * max( Q_current(next_state, action_i) ) 
 ```

 More at:
  * ...

 See also [A], [Gamma Discount Rate], [State Action Pair] 


# AlphaCode Model

 A LLM used for generating code. Built by the [DeepMind]. An alternative to the [Codex Model] built by [OpenAI]

 {% youtube "https://www.youtube.com/watch?v=t3Yh56efKGI" %}

 More at:
  * [https://www.deepmind.com/blog/competitive-programming-with-alphacode](https://www.deepmind.com/blog/competitive-programming-with-alphacode)

 See also [A], [Codex Model]


# AlphaFault

 [AlphaFold Model] does not know physics, but just do pattern recognition/translation.

 More at:
  * [https://phys.org/news/2023-04-alphafault-high-schoolers-fabled-ai.html](https://phys.org/news/2023-04-alphafault-high-schoolers-fabled-ai.html) 
  * paper - [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282689](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282689)

 See also [A], ...


# AlphaFold Model

 AlphaFold is an [artificial intelligence (AI)][AI] program developed by [DeepMind], a subsidiary of Alphabet, which performs predictions of protein structure. The program is designed as a deep learning system.

 AlphaFold AI software has had two major versions. A team of researchers that used AlphaFold 1 (2018) placed first in the overall rankings of the 13th [Critical Assessment of Structure Prediction (CASP)][CASP Challenge] in December 2018. The program was particularly successful at predicting the most accurate structure for targets rated as the most difficult by the competition organisers, where no existing template structures were available from proteins with a partially similar sequence.

 A team that used AlphaFold 2 (2020) repeated the placement in the CASP competition in November 2020. The team achieved a level of accuracy much higher than any other group. It scored above 90 for around two-thirds of the proteins in CASP's global distance test (GDT), a test that measures the degree to which a computational program predicted structure is similar to the lab experiment determined structure, with 100 being a complete match, within the distance cutoff used for calculating GDT.

 AlphaFold 2's results at CASP were described as "astounding" and "transformational." Some researchers noted that the accuracy is not high enough for a third of its predictions, and that it does not reveal the mechanism or rules of protein folding for the protein folding problem to be considered solved. Nevertheless, there has been widespread respect for the technical achievement.

 On 15 July 2021 the AlphaFold 2 paper was published at Nature as an advance access publication alongside open source software and a searchable database of species proteomes.

 But recently [AlphaFault] !

 {% pdf "{{site.assets}}/a/alphafold_model_paper.pdf" %}

 More at:
  * nature paper - [https://www.nature.com/articles/s41586-021-03819-2](https://www.nature.com/articles/s41586-021-03819-2)
  * [https://en.wikipedia.org/wiki/AlphaFold](https://en.wikipedia.org/wiki/AlphaFold)
  * [https://alphafold.com/](https://alphafold.com/)
  * online database - [https://alphafold.ebi.ac.uk/faq](https://alphafold.ebi.ac.uk/faq)
  * colab - [https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)

 See also [A], [AlphaGo Model], [AlphaZero Model]


# AlphaFold Protein Structure Database

 The protein structure database managed by [DeepMind] where all the protein structures predicted by the [AlphaFold Model] are stored.

 More at:
  * [https://alphafold.com/](https://alphafold.com/)

 See also [A}, ...


# AlphaGo Model

 AlphaGo was built by [DeepMind]. AI to play GO. Used reinforcement learning.

 {% youtube "https://www.youtube.com/watch?v=WXuK6gekU1Y" %}

 More at:
  * ...

 See also [A], [AlphaFold Model], [AlphaZero Model], [Reinforcement Learning]


# AlphaStar Model

 AlphaStar was built by [DeepMind]. Plays StarCraft II

 {% youtube "https://www.youtube.com/watch?v=jtlrWblOyP4" %}

 More at:
  * [https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning](https://www.deepmind.com/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning)
  * [https://www.nature.com/articles/s41586-019-1724-z.epdf](https://www.nature.com/articles/s41586-019-1724-z.epdf)

 See also [A], [OpenAI Five Model]


# AlphaTensor Model

 Better algorithm for tensor multiplication (on GPU ?). Based on AlphaZero. Built by [DeepMind]

 {% pdf "{{site.assets}}/a/alphatensor_nature_paper.pdf" %}

 More at:
  * announcement - [https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor)
  * paper in nature - [https://www.nature.com/articles/s41586-022-05172-4](https://www.nature.com/articles/s41586-022-05172-4)
  * github code - [https://github.com/deepmind/alphatensor](https://github.com/deepmind/alphatensor)
  * [https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/](https://venturebeat.com/ai/deepmind-unveils-first-ai-to-discover-faster-matrix-multiplication-algorithms/)

 See also [A], [AlphaZero Model], ...


# AlphaZero Model

 AI to play chess (and go and tensor algorithm). Built by [DeepMind]

 See also [A], [AlphaGo Model], [AlphaTensor Model], [MuZero Model]


# Amazon Company

 See also [A], [Amazon Web Services]


# Amazon Web Services

# AWS

 A subsidiary of [Amazon]

  * [AWS Bedrock] - rival to ChatGPT and DALL-E, foundation models for generative AI
  * [AWS Lex]
  * [AWS Polly] - Text to speech service
  * [AWS Recognition]

 See also [A], ...


# Ameca Robot

 {% youtube "https://www.youtube.com/watch?v=rxhzBiC-Q5Y" %}

 See also [A], [Robot]


# Andrew Ng Person

 {% youtube "https://www.youtube.com/watch?v=BY9KV8uCtj4" %}

 {% youtube "https://www.youtube.com/watch?v=vStJoetOxJg" %}

 More at:
  * [https://en.wikipedia.org/wiki/Andrew_Ng](https://en.wikipedia.org/wiki/Andrew_Ng)
  * deeplearning AI youtube channel - [https://www.youtube.com/@Deeplearningai](https://www.youtube.com/@Deeplearningai)

 See also [A], [People]


# Anomaly Detection

 Any deviation from a normal behavior is abnormal :warning: Evolution over time.

 See also [A], [Clustering], [Reconstruction Error], [Distance Methods], [X Density Method]


# Anthropic Company

 Models:
  * [Claude Model]

 More at:
  * home - [https://anthropic.ai/](https://anthropic.ai/)

 See also [A], ...


# Apache MXNet

 See also [A], [TensorFlow ML Framework]


# Apache Spark

 (with spark Sagemaker estimator interface?)


# Apple Company

 * [CoreML Framework] with the easy to get started [CreateML Application]
 * [Siri Virtual Assistant]

 See also [A], [Company]


# Apprentice Learning

 See also [A], [Stanford Autonomous Helicopter]


# Apriori Algorithm

 ~ a type of [unsupervised learning] in sub-class [association rule learning] used to ...

 The Apriori algorithm is a classic algorithm in data mining for learning association rules between items in a transactional database. Here are some key points about Apriori:

  * It employs a "bottom up" approach to find frequent itemsets through iterative passes over the dataset.
  * It starts by identifying individual items that satisfy minimum support.
  * In each subsequent pass, it combines the frequent items from the previous pass to form candidate itemsets.
  * Itemsets that still satisfy minimum support after a pass form the frequent itemsets.
  * The algorithm terminates when no new frequent itemsets are found in a pass.
  * Apriori uses the "downward-closure" property, which states that if an itemset is infrequent, its supersets should not be generated/tested.
  * It generates candidate itemsets efficiently by only exploring frequent subsets, pruning away infrequent candidates.
  * After frequent itemsets are found, association rules are generated based on minimum confidence constraints.
  * Performance degrades if there are lots of frequent itemsets or long patterns due to costly candidate generation.

 In summary, Apriori is an influential algorithm for mining associations that uses iterative passes over data and pruning strategies to efficiently discover frequent itemsets from which association rules can be derived.

 {% youtube "https://www.youtube.com/watch?v=T3Pd_3QP9J4" %}

 More at:
  * [https://pianalytix.com/association-rules-ml-method/](https://pianalytix.com/association-rules-ml-method/)

 See also [A], [Recommendation Engine]


# Arcade Learning Environment

# ALE

 The Arcade Learning Environment (ALE) is an extension of the [Atari Learning Environment] (ALE) that encompasses a broader range of arcade games beyond just the Atari 2600. While ALE focuses exclusively on Atari 2600 games, ALE expands the scope to include various arcade games from different platforms.

 ALE provides a unified framework and API for RL agents to interact with arcade games. It offers support for multiple game platforms, such as Atari 2600, MAME (Multiple Arcade Machine Emulator), and others. ALE provides a consistent interface for RL agents, allowing them to observe game screens, receive rewards, and take actions in a variety of arcade game environments.

 The key difference between the Arcade and Atari Learning Environments is the inclusion of additional arcade games in ALE. While the Atari LE is limited to Atari 2600 games, Arcade LE extends the game selection to include a broader range of arcade titles. This expansion increases the diversity of environments available for RL research and allows for more comprehensive evaluation of RL algorithms.

 {% pdf "https://arxiv.org/pdf/1207.4708.pdf" %}

 More at:
  * [https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
  * paper - [https://arxiv.org/abs/1207.4708](https://arxiv.org/abs/1207.4708)

 See also [A], ...


# Area Under The Curve

# AUC

 `~ helpful measurement to compare the classification performance of various models. The bigger the AUC, the better the model!"`.

 The curve is the Receiver Operating Characteristic (ROC) Curve][ROC Curve]! The area under the ROC curve (AUC) is a measure of the classifier's overall performance, with a value of 1 indicating perfect performance and a value of 0.5 indicating a performance no better than random guessing (ROC curve is diagonal ==> ...) .

 ![]( {{site.assets}}/a/area_under_the_curve.png){: width="100%" }

 ![]( {{site.assets}}/a/area_under_the_curve_code.png){: width="100%" }

 See also [A], ...


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


# Argparse Module

 ```
parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()
dataset = Dataset(args)
model = Model(dataset)
train(dataset, model, args)
print(predict(dataset, model, text='Knock knock. Whos there?'))
```

 More at:
  * [https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial](https://closeheat.com/blog/pytorch-lstm-text-generation-tutorial)

 See also [A], ...


# Artificial General Intelligence

# AGI

 AGI is the idealised solution many conceive when thinking about AI. While researchers work on the narrow and superficial, they talk about AGI, which represents the single story of AI, dating back to the 1950s, with a revival in the past decade. AGI implies two things about a solution that should not apply to business-centric problem-solving. First, a program has the general aptitude for human intelligence (perhaps all human intelligence). Second, an AGI is a general problem solver or a blank slate meaning any knowledge of a problem is rhetorical and independent of a strategy to solve that problem. Instead, the knowledge depends on some vague, ill-defined aptitude relating to the multidimensional structure of natural intelligence. If that sounds ostentatious, it’s because it is. Examples:
  * RL can solve arbitrary problems within these environments

 ```
First, we believe the clearest framing of general intelligence is a system that can do anything a human can do in front of a computer. A foundation model for actions, trained to use every software tool, API, and webapp that exists, is a practical path to this ambitious goal, and ACT-1 is our first step in this direction.
# adept.ai
 ```
 
 Risks:
  * Intelligence [control][AI Control] and [alignment][AI Alignment]
  * [Intelligence explosion][AI Explosion]

 {% youtube "https://www.youtube.com/watch?v=wHiOKDlA8Ac" %}

 {% youtube "https://www.youtube.com/watch?v=Mqg3aTGNxZ0" %}

 {% youtube "https://www.youtube.com/watch?v=qbIk7-JPB2c" %}

 {% pdf "https://arxiv.org/pdf/2303.12712.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/2303.12712](https://arxiv.org/abs/2303.12712)
  * risks - [https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence](https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence)

 See also [A], [Artificial Narrow Intelligence], [Artificial Super Intelligence]


# Artificial Intelligence

# AI

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
  3. :warning: Best definition found =  `AI is the science and engineering ...`
   * `... to use artificial devices`
    * current computer hardware and software, sensors, actuators, etc
   * `... to exhibit human capabilities`
    * perception - undertanding of data
    * cognition - reasoning and learning
     * action - execution and interaction
   * `... to solve problems addressed by humans`

 ![]( {{site.assets}}/a/artificial_intelligence_sensing_effecting.png){: width="100%" }

 ![]( {{site.assets}}/a/artificial_intelligence_machine_learning.png){: width="100%" }

 ![]( {{site.assets}}/a/artificial_intelligence_3_types.png){: width="100%" }

 More at:
  * state of AI
    * 2022 - [https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-in-2022-and-a-half-decade-in-review](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-in-2022-and-a-half-decade-in-review)

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


# Artificial Intelligence Challenge

# AI Challenge

 * 1994 - 2022+ : First [CASP Challenge], CASP13 in 2018 was won by [AlphaFold 1][AlphaFold Model]
 * 1997 - 1998 : [Deep Blue Challenge]
 * 2004 - 2005 : [DARPA Grand Challenge]
 * 2006 - 2009 : [Netflix Prize], in 2009 10% threshold was achieved, but customer data leak reported!
 * 2007 : [DARPA Urban Challenge]
 * 2011 - 2012 : [ImageNet Large Scale Visual Recognition Challenge]
 * 2015 : [DARPA Robotics Challenge]
 * 2021 : [DARPA Subterranean Challenge]

 See also [A], ...


# Artificial Intelligence Complete

# AI Complete

 Relates to NP complete from complexity.

 See also [A], [AI Hard]


# Artificial Intelligence Hard

# AI Hard

 Relates to NP hard from complexity.

 See also [A], [AI Complete]


# Artification Intelligence Risk Management Framework

# AI RMF

 Written by [NIST]

 {% pdf "{{site.assets}}/a/artificial_intelligence_risk_management_framework.pdf" %}

 More at:
  * [https://aibusiness.com/responsible-ai/microsoft-offers-support-on-responsible-ai-deployments](https://aibusiness.com/responsible-ai/microsoft-offers-support-on-responsible-ai-deployments)

 See also [A], ...


# Artificial Narrow Intelligence

# ANI

 ANI is often conflated with weak artificial intelligence. John Searle, philosopher and professor at the University of California, explained in his seminal 1980 paper, “Minds, Brains, and Programs,” that weak artificial intelligence would be any solution that is both narrow and a superficial look-alike to intelligence. Searle explains that such research would be helpful in testing hypotheses about segments of minds but would not be minds.[3] ANI reduces this by half and allows researchers to focus on the narrow and superficial and ignore hypotheses about minds. In other words, ANI purges intelligence and minds and makes artificial intelligence “possible” without doing anything. After all, everything is narrow, and if you squint hard enough, anything is a superficial look-alike to intelligence.

 {% pdf "{{site.assets}}/a/artificial_narrow_intelligence_paper.pdf" %}

 See also [A], [Artificial General Intelligence], [Artificial Super Intelligence]


# Artificial Neural Network

# ANN

 ~ `Can discover and approximate a(ny?) function given fixed(-count?) inputs and fixed(-count?) outputs! = universal function approximator` A multi-layer perceptron. Also known as Artificial Neural Network (ANN). Can be used in supervised or unsupervised learning.

 ![]( {{site.assets}}/a/artificial_neural_network_example.png ){: width="100%"}

 `For a young person, the university he/she is coming from is important. Compare that to someone with experience, whose university may not matter so much. but experience and field are important. ==> the weight of university is function of the age ==> That's what the second layer correct for!`

 ![]( {{site.assets}}/a/artificial_neural_network_types.png ){: width="100%"}


 The way for researchers to build an artificial brain or [neural network] using [artificial neurons][Artificial Neuron]. There are several types of ANN, including:
  * [Convolutional Neural Network] - used for [computer vision], such as [image recognition], [object detection], [image segmentation]. Use [filters][Image Kernel] to extract features from input image.
  * [Feedforward Neural Network] - [input][Input Layer], [hidden][Hidden Layer], and [output][Output Layer] layers. Information flows only in one direction.
  * [Recurrent Neural Network][RNN] - for sequential data, such as [natural language processing][NLP] and [speech recognition]. Have a feedback loop that allows information to flow in both directions.
  * [Generative Adversarial Network] - 2 neural networks, one that generate data, another tries to distinguish between real and fake data.
  * [Long Short-Term Memory Network] - a type of [RNN] that can remember long-term dependencies in sequential data. Used for [speech recognition] and [language translation]
  * [Autoencoder] - used for [unsupervised learning], where goal is to learn a compressed representation of the input data. The [encoder] compresses, the [decoder] reconstructs the original data.
  * [Deep belief network] - composed of multiple layers of [Restricted Boltzmann Machines]. Used for unsupervised learning and fine-tuned for [supervised learning] tasks.
  * ...

 The "knowledge/skills" of the ANN are encoded in their [parameters][Parameter]

 Hyperparameters:
  * Number of neurons per layer
  * Number of layers
  * [Activation function]
  * [Dropout function]

 {% youtube "https://www.youtube.com/watch?v=hfMk-kjRv4c" %}

 More at:
  * playground - [https://playground.tensorflow.org/](https://playground.tensorflow.org/)

 See also [A], [Hidden Layer], [Input Layer], [Output Layer], [Perceptron], [Universal Function Approximator]


# Artificial Neuron

 aka Node, currently artificial neurons are implemented as a [Perceptrons][Perceptron].

 Several (binary) input channels, to produce one output (binary value) that can be faned out. Input weights, Bias (add an offset vector for adjustment to prior predictions), non-linear activation function (sum+bias must meet or exceed activation threshold).

 ![]( {{site.assets}}/a/artificial_neuron.png ){: width="100%"}

 See also [A], [Activation Function], [Artificial Neuron Bias], [Input Weight]


# Artificial Neuron Bias

 A threshold the output needs to exceed for the output to fire

 See also [A], [Bias]


# Artificial Super Intelligence

# ASI

 ASI is a by-product of accomplishing the goal of AGI. A commonly held belief is that general intelligence will trigger an “intelligence explosion” that will rapidly trigger super-intelligence. It is thought that ASI is “possible” due to recursive self-improvement, the limits of which are bounded only by a program’s mindless imagination. ASI accelerates to meet and quickly surpass the collective intelligence of all humankind. The only problem for ASI is that there are no more problems. When ASI solves one problem, it also demands another with the momentum of Newton’s Cradle. An acceleration of this sort will ask itself what is next ad infinitum until the laws of physics or theoretical computation set in. The University of Oxford scholar Nick Bostrom claims we have achieved ASI when machines have more intelligent than the best humans in every field, including scientific creativity, general wisdom, and social skills. Bostrom’s depiction of ASI has religious significance. Like their religious counterparts, believers of ASI even predict specific dates when the Second Coming will reveal our savior. Oddly, Bostrom can’t explain how to create artificial intelligence. His argument is regressive and depends upon itself for its explanation. What will create ASI? Well, AGI. Who will create AGI? Someone else, of course. AI categories suggest a false continuum at the end of which is ASI, and no one seems particularly thwarted by their ignorance. However, fanaticism is a doubtful innovation process.

 See also [A], [Artificial General Intelligence], [Artificial Narrow Intelligence]


# Association for the Advancement of Artificial Intelligence

# AAAI

 Founded in 1979, the Association for the Advancement of Artificial Intelligence (AAAI) (formerly the American Association for Artificial Intelligence) is a nonprofit scientific society devoted to advancing the scientific understanding of the mechanisms underlying thought and intelligent behavior and their embodiment in machines. AAAI aims to promote research in, and responsible use of, artificial intelligence. AAAI also aims to increase public understanding of artificial intelligence, improve the teaching and training of AI practitioners, and provide guidance for research planners and funders concerning the importance and potential of current AI developments and future directions.

 AAAI’s goals are:
  * Promoting research in, and responsible use of, artificial intelligence (AI)
  * Increasing public understanding of artificial intelligence
  * Improving the teaching and training of AI practitioners
  * Providing guidance for research planners and funders concerning the importance and potential of current AI developments and future directions.

 AAAI’s activities include:

  * Organizing and sponsoring conferences, symposia, and workshops
  * Publishing a quarterly magazine for all members
  * Publishing a series of proceedings, including the annual proceedings for the AAAI Conference on Artificial Intelligence
  * Advocating for members throughout the world through educational programs and governmental outreach
  * Awarding grants and scholarships

 More at:
  * [https://aaai.org/about-aaai/](https://aaai.org/about-aaai/)
  * ai-topics - [https://aitopics.org/search](https://aitopics.org/search)
  * conferences
    * [https://www.aies-conference.com/](https://www.aies-conference.com/)
    * [https://aaai-23.aaai.org/](https://aaai-23.aaai.org/)

 See also [A], ...


# Association Rule Learning

 ~ a type of [unsupervised learning] used to uncover relationships and associations between variables in transactional and relational datasets.

 Association rules are a technique in machine learning used to uncover relationships between variables in large datasets. The key concepts are:

  * Association rules identify associations between items in a [dataset]. For example, an association rule could identify that customers who purchase bread and butter together frequently also tend to purchase jam.
  * Associations are based on the frequency of items occurring together in transactions and how strongly they are associated.
  * Rules are generated by analyzing data for frequent if/then patterns and using measures of support and confidence to identify the most significant relationships.
  * Support refers to how frequently the items appear together in the data. Confidence refers to the probability of item Y appearing in transactions with item X. Rules must meet minimum support and confidence thresholds to be considered significant.
  * Market basket analysis is a key application of association rules for discovering associations in retail transaction data. But it has also been used for other applications like bioinformatics, intrusion detection, and web usage mining.
  * Algorithms like Apriori and FP-Growth are used to efficiently find association rules in large high-dimensional datasets.

 In summary, association rule learning is an unsupervised learning technique to uncover relationships and associations between variables in transactional and relational datasets. It relies on support and confidence statistical measures to identify the strongest associations.

 Metrics:
  * Support: Support: This Gives The Fraction Of Transactions Which Contains Item A And B. This Tells Us About The Frequently Bought Items Or The Combination Of Items Bought Frequently.  Support = Freq(A,B)N 
  * Confidence: It Tells Us How Often Items A And B Occur Together, Given The Number Of Times A Occurs. Confidence = Freq(A,B)Freq(A)
  * Lift: It Indicates The Strength Of A Rule Over The Random Occurrence Of A And B. This Tells Us The Strength Of Any Rule. Lift = Supportsupport(A) * Support(B)

 Algorithms:
  * [Apriori Algorithm]
  * [Equivalence Class Clustering And Bottom-Up Lattice Trasversal (ECLAT)][ECLAT]
  * [FP-Growth Algorithm]

 Use-cases:
  * Market basket analysis
  * ...

 See also [A], ...


# Asynchronous Advantage Actor-Critic Algorithm

# A3C Algorithm

 A [policy gradient algorithm] used in [reinforcement learning]

 {% youtube "https://www.youtube.com/watch?v=OcIx_TBu90Q" %}

 More at:
  * code - [https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py)
  * Articles
    * [https://www.neuralnet.ai/asynchronous-deep-reinforcement-learning/](https://www.neuralnet.ai/asynchronous-deep-reinforcement-learning/)


 See also [A], ...


# Atari Learning Environment

 The Atari Learning Environment (ALE) is an open-source software platform developed for research in reinforcement learning (RL). It is built upon the popular Atari 2600 video game console, which provides a diverse set of game environments for RL agents to interact with. ALE allows researchers to develop and evaluate RL algorithms by providing a standardized interface and a collection of Atari 2600 games as benchmark tasks.

 The ALE provides a set of APIs (Application Programming Interfaces) that enable RL agents to interact with the Atari games. Agents can observe the game screen, receive reward signals based on their actions, and make decisions on how to act in the game environment. ALE also provides a scoring system that allows for performance comparison across different algorithms and agents.

 The primary objective of ALE is to facilitate the development and evaluation of RL algorithms by providing a common framework and standardized benchmark tasks. It has been widely used in the research community to test and compare various RL algorithms and techniques.

 {% pdf "https://arxiv.org/pdf/1312.5602v1.pdf" %}

 More at:
  * integration with the [Arcade Learning Environment]
  * integration with OpenAI Gym - [https://www.gymlibrary.dev/environments/atari/index.html](https://www.gymlibrary.dev/environments/atari/index.html)
  * paper - 

 See also [A], ...


# Atlas Robot

 Atlas is a bipedal humanoid [robot] primarily developed by the American robotics company [Boston Dynamics] with funding and oversight from the [U.S. Defense Advanced Research Projects Agency (DARPA)][DARPA]. The robot was initially designed for a variety of search and rescue tasks, and was unveiled to the public on July 11, 2013.

 {% youtube "https://www.youtube.com/watch?v=rVlhMGQgDkY" %}

 More at:
  * [https://www.bostondynamics.com/atlas](https://www.bostondynamics.com/atlas)
  * [https://en.wikipedia.org/wiki/Atlas_%28robot%29](https://en.wikipedia.org/wiki/Atlas_%28robot%29)

 See also [A], ...


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

 {% youtube "https://www.youtube.com/watch?v=g2BRIuln4uc" %}

 See also [A], [Attention], [Attention Score], [Autoencoding], [Autoregressive], [BERT Model], [GPT Model], [Multi-Head Attention], [T5 Model]


# Attribute

 See also [A], [Negative Attribute], [Positive Attribute]]


# Autoencoder

 Let’s now discuss autoencoders and see how we can use neural networks for dimensionality reduction. The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding-decoding scheme using an iterative optimisation process. So, at each iteration we feed the autoencoder architecture (the encoder followed by the decoder) with some data, we compare the encoded-decoded output with the initial data and backpropagate the error through the architecture to update the weights of the networks. Thus, intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part of the information can go through and be reconstructed. Looking at our general framework, the family E of considered encoders is defined by the encoder network architecture, the family D of considered decoders is defined by the decoder network architecture and the search of encoder and decoder that minimise the reconstruction error is done by gradient descent over the parameters of these networks.

 ![]( {{site.assets}}/a/autoencoder.png ){: width="100%"}

 See also [A], [Autoencoding], [Backpropagation], [Decoder], [Denoising Autoencoder], [Dimensionality Reduction], [Disentangled Variational Autoencoder], [Encoder], [Encoder-Decoder Model], [Hidden State], [Linear Autoencoder], [Unsupervised Deep Learning Model], [Unsupervised Learning], [Variational Autoencoder]


# AutoGPT  Model

 {% youtube "https://www.youtube.com/watch?v=LqjVMy2qhRY" %}

 {% youtube "https://www.youtube.com/watch?v=0m0AbdoFLq4" %}

 Derivative versions:
  * AgentGPT
    * site - [https://agentgpt.reworkd.ai/](https://agentgpt.reworkd.ai/)
    * code - [https://github.com/reworkd/AgentGPT](https://github.com/reworkd/AgentGPT)
  * God Mode
    * site - [https://godmode.space/](https://godmode.space/)

 More at:
  * home - [https://news.agpt.co/](https://news.agpt.co/)
  * code - [https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)
  * wikipedia - [https://en.wikipedia.org/wiki/Auto-GPT](https://en.wikipedia.org/wiki/Auto-GPT)
  * langchain - [https://python.langchain.com/en/latest/use_cases/autonomous_agents.html](https://python.langchain.com/en/latest/use_cases/autonomous_agents.html)
  * UI
   * huggingface - [https://huggingface.co/spaces/aliabid94/AutoGPT](https://huggingface.co/spaces/aliabid94/AutoGPT)
   * [https://generativeai.pub/autogpt-now-supports-web-ui-heres-how-you-can-try-fd94b2a6ddad](https://generativeai.pub/autogpt-now-supports-web-ui-heres-how-you-can-try-fd94b2a6ddad)
  * articles
   * [https://www.zdnet.com/article/what-is-auto-gpt-everything-to-know-about-the-next-powerful-ai-tool/](https://www.zdnet.com/article/what-is-auto-gpt-everything-to-know-about-the-next-powerful-ai-tool/)

 See also [A], ...


# Automatic Speech Recognition

# ASR

 Possible thanks to [Recurrent Neural Network] such as [LSTM Network]

 See also [A], ...


# Automation

 Automation refers to the use of technology, machinery, or systems to perform tasks or processes with minimal human intervention. It involves the implementation of control systems, sensors, and algorithms to carry out repetitive or complex actions automatically, reducing the need for manual effort. Automation aims to improve efficiency, productivity, accuracy, and reliability by streamlining operations and reducing human error. It can be applied in various domains, including manufacturing, transportation, agriculture, healthcare, and information technology. Examples of automation include robotic assembly lines, automated customer service systems, self-driving cars, and smart home devices.

 See also [A], ...


# Autonomous Vehicle

 See also [DARPA Grant Challenge], [DARPA Urban Challenge]


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

 ![]( {{site.assets}}/a/attention_score.png ){: width="100%"}

 Multi-Head Attention consists of several attention layers running in parallel. The Attention layer takes its input in the form of three parameters, known as the Query, Key, and Value (aka Q,K,V). All three parameters are similar in structure, with each word in the sequence represented by a vector. In transformers is used for encoder and decoder.

 ![]( {{site.assets}}/m/multi_head_attention.png ){:width="100%"}

 ![]( {{site.assets}}/a/attention_score_formula.png ){:width="100%"}

 {% youtube "https://www.youtube.com/watch?v=_UVfwBqcnbM" %}

 {% youtube "https://www.youtube.com/watch?v=4Bdc55j80l8" %}

 More at :
  * [https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)
  * [https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

 See also [A], [Attention], [Attention-Based Model], [Multi-Head Attention], [Positional Encoding], [Transformer Model]


# Augmented Language Model

# ALM

 A language model that can use external tools.

 LLM reasons to call an external tool, gets halted to fetch the tool’s response as observation, and then decides the next action based on all preceding responses. This technique is also sometimes referred as Augmented Language Models (ALMs).

 More at:
  * [https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/](https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/)

 See also [A], [ReACT Prompting]


# Augmented Reality

# AR

 Augmented reality (AR) is an interactive experience that combines the real world and computer-generated content. The content can span multiple sensory modalities, including visual, auditory, haptic, somatosensory and olfactory. AR can be defined as a system that incorporates three basic features: a combination of real and virtual worlds, real-time interaction, and accurate 3D registration of virtual and real objects. The overlaid sensory information can be constructive (i.e. additive to the natural environment), or destructive (i.e. masking of the natural environment). This experience is seamlessly interwoven with the physical world such that it is perceived as an immersive aspect of the real environment. In this way, augmented reality alters one's ongoing perception of a real-world environment, whereas virtual reality completely replaces the user's real-world environment with a simulated one.

 Augmented reality is largely synonymous with [Mixed Reality].

 {% youtube "https://www.youtube.com/watch?v=Zu1p99GJuTo" %}

 More at:
  * [https://en.wikipedia.org/wiki/Augmented_reality](https://en.wikipedia.org/wiki/Augmented_reality)

 See also [A], [Virtual Continuum]


# Autoencoder Bottleneck

 This is the name of the hidden layer with the least neuron in an autoencoder architecture. This is where the information is compressed, encoded in the latent space!

 ![]( {{site.assets}}/a/autoencoder_bottleneck.png ){:width="100%"}

 See also [A], [Autoencoder], [Latent Space]


# Autoencoder Type

 There are 2 types of autoencoders:
  * input X --> Latent representation
  * input X --> Latent distribution

 ![]( {{site.assets}}/a/autoencoder_type.png){: width="100%"}

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

 ![]( {{site.assets}}/a/automation_sae_j3016.png ){:width="100%"}

 More at:
  * [https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic](https://www.sae.org/news/2019/01/sae-updates-j3016-automated-driving-graphic)

 See also

# AutoML

 significant advances in automated machine learning that can “automatically discover complete machine learning algorithms just using basic mathematical operations as building blocks.”

 See also [A], [Hyperparameter Tuning], [Early Stopping], [Neural Architecture Search]

# Autoregressive Model

 Goal is to predict a future token (word) given either the past tokens or the future tokens but not both. (If both --> auto-encoding). Autoregressive models such as decoders are iterative and reused their temporary, incomplete output to generate the next, more complete output. Iterations stop when encoder input is exhausted (?). Well-known autoregressive models/use-cases are:
  * Predicting next work in a sentence (auto-complete)
  * Natural language generation
  * [GPT Models][GPT Model]
 ```
If you don't ____ (forward prediction)
____ at the sign, you will get a ticket (backward prediction)
 ```

 See also [A], [Autoencoding], [Casual Language Modeling], [Decoder]


# Autoregressive Convolutional Neural Network

# AR-CNN

 `~ representing the problem/solution as a time series of images`. Q: How do you generated the images? Q: How many images? Ex: music --> given the previous images (piano roll with notes until now), find the next image (piano roll with new note!) ? NO, WRONG!! More like imagine your first COMPLETE draft, you review the draft, you review it once more, and again, getting better and better each time until it cannot get better anymore! `Each action (i.e. addition or removal of exactly one note) is a transformation from one piano roll image to another piano roll image`.

 See also [A], [Autoregressive Model], [Convolutional Neural Network]


# Autoregressive Model

# AR Model

 `~ analyze the past steps (or future but not both) to identify the next step = learn from the past iteration (or future but not both) ONLY`. Unlike the GANs approach described before, where music generation happened in one iteration, autoregressive models add notes over many iterations. The models used are called autoregressive (AR) models because the model generates music by predicting future music notes based on the notes that were played in the past. In the music composition process, unlike traditional time series data, one generally does more than compose from left to right in time. New chords are added, melodies are embellished with accompaniments throughout. Thus, instead of conditioning our model solely on the past notes in time like standard autoregressive models, we want to condition our model on all the notes that currently exist in the provided input melody. For example, the notes for the left hand in a piano might be conditioned on the notes that have already been written for the right hand.

 See also [A], [Autoregressive Convolutional Neural Network], [Generative Adversarial Network], [GPT Model], [Time-Series Predictive Analysis]


# AWS Bedrock

 A [ChatGPT][ChatGPT Model] and [DALL-E][DALL-E Model] rival offered by [Amazon Web Services]

 More at:
  * [https://aws.amazon.com/bedrock/](https://aws.amazon.com/bedrock/)
  * [https://www.businessinsider.com/amazon-bedrock-aws-ai-chatgpt-dall-e-competitor-2023-4](https://www.businessinsider.com/amazon-bedrock-aws-ai-chatgpt-dall-e-competitor-2023-4)

 See also [A], ...

# AWS Lex

 Text or Speech conversation.

 See also [A], [Amazon Web Services]


# AWS Polly

 Lifelike speech. Text to speech.

 See also [A], [Amazon Web Services]


# AWS Recognition

 Used for image analysis.

 See also [A], [Amazon Web Services]
