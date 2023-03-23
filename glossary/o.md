---
title: O
permalink: /o/

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


# Object Detection

 More at:
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [O], [Computer Vision], [Convoluted Neural Network]


# Object Recognition

 See also [O], [Computer Vision]


# Object Tracking

 More at:
  * [https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b](https://heartbeat.fritz.ai/the-5-computer-vision-techniques-that-will-change-how-you-see-the-world-1ee19334354b)

 See also [O], [Computer Vision]


# Observation

 In Reinforcement learning, an observation using one or more sensor (ex: camera) can help you identify in which state you are.

 See also [O], [Reinforcement Learning], [State]


# One-Cold Encoding

 Same as one-hot encoding except the 1 is a 0 and 0s are 1s. Example yellow = [1, 0, 1, 1].

 See also [O], [One-Hot Encoding]


# One-Hot Encoding

 Categorical data refers to variables that are made up of label values, for example, a “color” variable could have the values “red“, “blue, and “green”. Think of values like different categories that sometimes have a natural ordering to them. Some machine learning algorithms can work directly with categorical data depending on implementation, such as a decision tree, but most require any inputs or outputs variables to be a number, or numeric in value. This means that any categorical data must be mapped to integers. One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one-hot, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns. Each integer value is represented as a binary vector. All the values are zero, and the index is marked with a 1.
 
 ```
red    = [1, 0, 0, 0]           <== vector !
yellow = [0, 1, 0, 0]
blue   = [0, 0, 1, 0]
green  = [0, 0, 0, 1]
 ```

 with pandas
 
 ```
import pandas as pd

df = pd.DataFrame({"col1": ["Sun", "Sun", "Moon", "Earth", "Moon", "Venus"]})
df_new = pd.get_dummies(df, columns=["col1"], prefix="Planet")
print(df)

#     Planet_Earth  Planet_Moon  Planet_Sun  Planet_Venus
# 0             0            0           1             0                <== Sun
# 1             0            0           1             0                <== Sun
# 2             0            1           0             0                <== Moon
# 3             1            0           0             0                <== Earth
# 4             0            1           0             0                <== Moon
# 5             0            0           0             1                <== Venus
 ```

 See also [O], [One-Cold Encoding]


# One-Shot Learning

 In addition to the task description, the model sees a single example of the task. No gradient updates are performed.

 ```
Translate English to French:                # Task description
sea otter => loutre de mer                  # Example
cheese =>                                   # Prompt
 ```

 ~ `not enough data --> encoder + similarity function?` Ex: I only have 1 picture of the employee, how can I recognize people coming in as an employee or not? Treat as a classification problem and each class correspond to 1 employee. How do you learn to train a classifier based on 1 sample? Better to look at this problem as a nearest-neighbor problem! Question: How do you detect similarities? Use a similarity function, an encoder! So sequence of (1) an encoder and (2) a nearest neighbor algorithm.

 ![]( {{site.assets}}/o/one_shot_learning.png ){: width="100%"}

 Few-, one-, and zero-shot settings are specialized cases of zero-shot task transfer. In a few-shot setting, the model is provided with a task description and as many examples as fit into the context window of the model. In a one-shot setting, the model is provided with exactly one example and, in a zero-shot setting, with no example.

 See also [O], [Encoder], [Few-Shot Learning], [Siamese Network], [Similarity Function], [Transfer Learning], [Zero-Shot Learning], [Zero-Shot Task Transfer]


# OpenAI Company

 [Microsoft][Micosoft Company] + [OpenAI][OpenAI Company] ~ [Google][Google Company] + [DeepMind][DeepMind Company]

 Models:
  * [ChatGPT][ChatGPT Model]: An fine-tuned model of GTP that is based on dialog
  * [CLIP][CLIP Model]: A model that can put a legend to an image
  * [Codex][Codex Model]: A [LLM] that specialize on generating code
  * [DALL-E][DALL-E Model]: A [Diffusion Model] that from text can generate images
  * [Five][OpenAI Five Model]: An agent that isnow world champion at the Dota2 game!
  * [GPT][GPT Model]: A Generative model for text
  * [Gym Environment][OpenAI Gym Environment]: Environments for development of [Reinforcement Learning] algorithms.
  * [Whisper][Whisper Model]: A text-to-speech model

 More at:
  * [http://www.openai.com](http://www.openai.com)

 See also [O], [Microsoft Company]


# OpenAI Five Model

 5 agents (characters) that work together in collaboration! 
 Incomplete information

 {% youtube "https://www.youtube.com/watch?v=tfb6aEUMC04" %}

 More at:
  * [https://openai.com/research/openai-five](https://openai.com/research/openai-five)
  * [https://openai.com/blog/openai-five-benchmark](https://openai.com/blog/openai-five-benchmark)
  * [https://www.twitch.tv/videos/293517383](https://www.twitch.tv/videos/293517383)
  * [https://openai.com/blog/openai-five-finals](https://openai.com/blog/openai-five-finals)

 See also [O], [AlphaStar Model]


# OpenAI Gym Environment

 See also [O], [OpenAI Company], [Rocket League Gym]


# OpenCV Library

 See also [O], [Computer Vision]


# Optimizer

 To minimize the prediction error or loss, the model while experiencing the examples of the training set, updates the model parameters W. These error calculations when plotted against the W is also called cost function plot J(w), since it determines the cost/penalty of the model. So minimizing the error is also called as minimization the cost function. But how exactly do you do that? Using optimizers.
 
 ```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)
 ```

 More at:
  * [https://keras.io/api/optimizers/](https://keras.io/api/optimizers/)

 See also [O], [Loss Function]


# Output Layer

 See also [O], [Layer], [Neural Network]


# Output Perturbation

 We introduce gaussian noise in the output. Easy to implement as treat the model as a blackbox. But if people have access to the model itself, they can bypass noise the "addititor". In general, it is better if the noise is built in the model!

 See also [O], [Differential Privacy]


# Overfitting

 ~ a Model that is overly complex and leads to high variance and low bias = noise is memorized in model! . In contrast, a program that memorizes the training data by learning an overly-complex model could predict the values of the response variable for the training set accurately, but will fail to predict the value of the response variable for new examples. `A well functioning ML algorithm will separate the signal from the noise`. If the algorithm is too complex or flexible (e.g. it has too many input features or it’s not properly regularized), it can end up “memorizing the noise” instead of finding the signal. This overfit model will then make predictions based on that noise. It will perform unusually well on its training data… yet very poorly on new, unseen data. `A key challenge with overfitting, and with machine learning in general, is that we can’t know how well our model will perform on new data until we actually test it`. To address this, we can split our initial dataset into separate training and test subsets. To avoid over-fitting, try hyperparameter optimization.

 Reasons for overfitting:
  * too few training examples
  * running the training process for too many epochs

 ![]( {{site.assets}}/o/underfitting_overfitting_balanced.png ){: width="100%"}

 Beware:
  * Overfitting is responsible for the membership inference attack

 See also [O], [Balanced Fitting], [Bias], [Early Stopping], [Hyperparameter Optimization], [Membership Inference Attack], [Principal Component Analysis], [Test Subset], [Training Subset], [Underfitting], [Variance]


# Overtraining

 Overtraining is when a machine learning model can predict training examples with very high accuracy but cannot generalize to new data. This leads to poor performance in the field. Usually, this is a result of too little data or data that is too homogenous.

 See also [O], [Overfitting]
