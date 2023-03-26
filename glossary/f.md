---
title: F
permalink: /f/

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


# F1 Score

 A Measure of accuracy of a model. Used to find hyperparameter optimization.

 When to use? F1-Score is used when the False Negatives and False Positives are important. F1-Score is a better metric for Imbalanced Data.

 See also [F], [Confusion Matrix], [Hyperparameter Optimization]


# Facebook Company

 Models:
  * [CICERO][CICERO Model]: Strategy game with multiplayer interaction
  * [ESMFold][ESMFold Model]: Protein folding
  * [LLaMA][LLaMA Model]: Large Language Model open-sourced
  * [Make-A-Video][Make-A-Video Model]: Text to video model
  * [Pluribus][Pluribus Model]: Plays poker better than humans

 More at:
  * [https://github.com/facebookresearch](https://github.com/facebookresearch)

 See also [F], ...


# Feature

 `Input parameter to the model`. Which features are important to make the right prediction? Beware that the dataset needs to be at least 10 times the number of features.

 See also [F], [Data Point], [Dataset], [Explanatory Variable], [Feature Engineering], [Feature Extraction], [Feature Vector], [Naive Bayes]


# Feature Engineering

 Feature engineering is an iterative process that often requires experimentation and creating many models to find the input features that give the best model performance. You want features that are weakly correlated with each other, but that strongly correlated with the output.  Beware that the data-set needs to be at least 10 times the number of features. Example: for call routing in a call center
  * Use for feature the item that was last purchased
  * The date of the last purchase, or rather re-engineered the number of days since the last purchase
  * if the caller owns a kindle
  * etc
 Example for image recognition of a car
  * recognize the wheels
  * recognize the shape of the body
  * If a car has all of those features then it must be a car

 See also [F], [Dataset], [Feature], [Feature Vector]


# Feature Extractor

 Take an image as input and extract image kernels? Used in place of NLP tokenizer in vision extractor. Turn an image in another image with only important features/objects to be used in the captioning?

 See also [F], [Tokenizer], [Vision Transformer]


# Feature Extraction

 Feature extraction (from raw data) is a part of the dimensionality reduction process, in which, an initial set of the raw data is divided and reduced to more manageable groups. So when you want to process it will be easier. The most important characteristic of these large datasets is that they have a large number of variables. These variables require a lot of computing resources to process. So Feature extraction helps to get the best feature from those big datasets by selecting and combining variables into features, thus, effectively reducing the amount of data. These features are easy to process, but still able to describe the actual dataset with accuracy and originality. 

 See also [F], [Dimensionality Reduction], [Feature], [Principal Component Analysis]


# Feature Learning

 before classifier training!


# Feature Vector

 Single column matrix (a vector) that contains all the inputs to a model (ex: artificial neuron). 

 See also [F], [Feature], [Vector]


# Feedback

 Several ways to give feedback:
  * Two stars and a wish – two positive things about the presentation and one suggestion,
  * The 3, 2, 1, Formula – Three likes, Two suggestions, and One question.
  * Finally, feedback can be given based on things like the content of the presentation, use of visuals, eye contact, etc.
 
 See also [F], [Reward]


# Feedback-Based Learning

 Refine your behavior based on the feedback from the crowd.

 Questions:
  * How is this different from a reward in reinforcement learning? ==> selection bias ! I use the feedback I want! The goal is not to get the maximum reward, but to get to your destination which the reward model does not know about!.

 Example:
  * Ask people what they want, not a faster carriage, but a car

 More at:
  * RLHF is flawed? - [https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the](https://astralcodexten.substack.com/p/perhaps-it-is-a-bad-thing-that-the)
 
 See also [F], [Feedback], [Reinforcement Learning], [Reinforcement Learning Human Feedback]


# Feedforward Neural Network

 Output of a layer only feed the input of downstream layers. Input weights can be computed using backpropagation. This is the opposite of a Recurrent Neural Network. 

 See also [F], [Backpropagation], [Neural Network], [Recurrent Neural Network], [Softmax Function]


# Few-Shot Learning

 In addition to the task description the model sees a few examples of the task. No gradient updates are performed.

 ```
Translate English to French                # Task description
sea otter => loutre de mer                 # Example 1
peppermint => menthe poivree               # ...
plush girafe => girafe peluche             # Example N
cheese =>
 ```
 Deductions from a few hand-picked examples. ex: You watch someone playing a game. After he or she  played 5 round,, you say, oh yes, I think I can play the game. Few-, one-, and zero-shot settings are specialized cases of zero-shot task transfer. In a few-shot setting, the model is provided with a task description and as many examples as fit into the context window of the model. In a one-shot setting, the model is provided with exactly one example and, in a zero-shot setting, with no example. 

 See also [F], [One-Shot Learning], [Zero-Shot Task Transfer], [Zero-Shot Learning], 


# File Mode

 Each step of the training process generate file and are not streamed (?). 

 See also [F], [Pipe Mode]


# Fine-Tuning

 More at:
   * GPT fine-tuning - [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

 See [Supervised Fine-Tuning]


# Flow-Based Model

 More at :
  * [https://blog.kthais.com/flow-based-generative-models-a4de5024efcc](https://blog.kthais.com/flow-based-generative-models-a4de5024efcc)
  * Ahttps://en.wikipedia.org/wiki/Flow-based_generative_model](https://en.wikipedia.org/wiki/Flow-based_generative_model)
 
 See also [F], [Generative Model]


# Forbidden Planet Movie

 Forbidden Planet is a 1956 American science fiction film from Metro-Goldwyn-Mayer.

 Forbidden Planet pioneered several aspects of science fiction cinema. It was the first science fiction film to depict humans traveling in a faster-than-light starship of their own creation. It was also the first to be set entirely on another planet in interstellar space, far away from Earth. The Robby the Robot character is one of the first film robots that was more than just a mechanical "tin can" on legs; Robby displays a distinct personality and is an integral supporting character in the film.

 {% youtube "https://www.youtube.com/watch?v=Gu-peHw2w6I" %}

 More at:
  * [https://en.wikipedia.org/wiki/Forbidden_Planet](https://en.wikipedia.org/wiki/Forbidden_Planet) 

 See also [F], [AI Movie]


# Forest Of Stumps

 See also [F], [AdaBoost], [Decision Stump], [Gini Impurity Index], [Random Forest], [Weighted Gini Impurity Index]


# Fully Connected Layer

# FC Layer

 A List of feature values becomes a list of votes (which are weighted to following layers).

 ![]( {{site.assets}}/f/fully_connected_layer.png ){: width="100%}
 

 See also [F], [Convoluted Layer], [Convoluted Neural Network], [Poll Layer], [ReLU Layer]


# Function Estimation

 Here we are trying to predict a variable y given an input vector x. We assume that there is a function f(x) that describes the approximate relationship between y and x. For example, we may assume that y = f(x) + ε, where ε stands for the part of y that is not predictable from x. In function estimation, we are interested in approximating f with a model or estimate fˆ. Function estimation is really just the same as estimating a parameter θ; the function estimator fˆis simply a point estimator in function space. Ex: in polynomial regression we are either estimating a parameter w or estimating a function mapping from x to y. 

 See also [F], [Estimator], [Point Estimator]
