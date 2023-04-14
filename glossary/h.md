---
title: H
permalink: /h/

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


# Hallucination

 More at:
  * [https://venturebeat.com/ai/whats-next-in-large-language-model-llm-research-heres-whats-coming-down-the-ml-pike/](https://venturebeat.com/ai/whats-next-in-large-language-model-llm-research-heres-whats-coming-down-the-ml-pike/)

 See also [H], [Large Language Model]


# Hand Gesture Recognition

# HGR

 Gesture recognition provides real-time data to a computer to make it fulfill the user’s commands. Motion sensors in a device can track and interpret gestures, using them as the primary source of data input. A majority of gesture recognition solutions feature a combination of 3D depth-sensing cameras and infrared cameras together with machine learning systems. Machine learning algorithms are trained based on labeled depth images of hands, allowing them to recognize hand and finger positions.

 Gesture recognition consists of three basic levels:
  * Detection. With the help of a camera, a device detects hand or body movements, and a machine learning algorithm segments the image to find hand edges and positions.
  * Tracking. A device monitors movements frame by frame to capture every movement and provide accurate input for data analysis.
  * Recognition. The system tries to find patterns based on the gathered data. When the system finds a match and interprets a gesture, it performs the action associated with this gesture. Feature extraction and classification in the scheme below implements the recognition functionality.

 ![]( {{site.assets}}/h/hand_gesture_recognition.jpeg ){: width="100%"}

 Many solutions use vision-based systems for hand tracking, but such an approach has a lot of limitations. Users have to move their hands within a restricted area, and these systems struggle when hands overlap or aren’t fully visible. With sensor-based motion tracking, however, gesture recognition systems are capable of recognizing both static and dynamic gestures in real time.

 In sensor-based systems, depth sensors are used to align computer-generated images with real ones. Leap motion sensors are also used in hand tracking to detect the number and three-dimensional position of fingers, locate the center of the palm, and determine hand orientation. Processed data provides insights on fingertip angles, distance from the palm center, fingertip elevation, coordinates in 3D space, and more. The hand gesture recognition system using image processing looks for patterns using algorithms trained on data from depth and leap motion sensors:
  1. The system distinguishes a hand from the background using color and depth data. The hand sample is further divided into the arm, wrist, palm, and fingers. The system ignores the arm and wrist since they don’t provide gesture information.
  1. Next, the system obtains information about the distance from the fingertips to the center of the palm, the elevation of the fingertips, the shape of the palm, the position of the fingers, and so on.
  1. Lastly, the system collects all extracted features into a feature vector that represents a gesture. A hand gesture recognition solution, using AI, matches the feature vector with various gestures in the database and recognizes the user’s gesture.

 Depth sensors are crucial for hand tracking technology since they allow users to put aside specialized wearables like gloves and make HCI more natural.


 More at:
  * [https://intellias.com/hand-tracking-and-gesture-recognition-using-ai-applications-and-challenges/](https://intellias.com/hand-tracking-and-gesture-recognition-using-ai-applications-and-challenges/)

 See also [H], ...


# Hand Tracking

 ![]( {{site.assets}}/h/hand_tracking.gif )

 More at:
  * [https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html)

 See also [H], [Hand Gesture Recognition]


# Hanson Robotics Company

 Hanson Robotics Limited is a Hong Kong-based engineering and robotics company founded by David Hanson, known for its development of human-like robots with artificial intelligence (AI) for consumer, entertainment, service, healthcare, and research applications. The robots include Albert HUBO, the first walking robot with human-like expressions; BINA48, an interactive humanoid robot bust; and Sophia, the world's first robot citizen. The company has 45 employees.

 Hanson Robotics’ robots feature a patented spongy elastomer skin called Frubber that resembles human skin in its feel and flexibility. Underneath the Frubber are proprietary motor control systems by which the robots mimic human expressions.

 {% youtube "https://www.youtube.com/watch?v=2vAtJYRwegg" %}

 More at:
  * [https://www.hansonrobotics.com/](https://www.hansonrobotics.com/) 
  * [https://en.wikipedia.org/wiki/Hanson_Robotics](https://en.wikipedia.org/wiki/Hanson_Robotics)

 See also [H], [Company], [Sophia Robot]


# Hebbian Learning

 "Neurons that fire together, wire together" Linked to learning and neural mechanism. Repeated or persistent firing changes synaptic weight due to increased efficiency. Synaptic modifications can be hebbian, anti-hebbian, or non-hebbian.

 More at :
  * [https://anthony-tan.com/Supervised-Hebbian-Learning/](https://anthony-tan.com/Supervised-Hebbian-Learning/)

 See also [H], 


# Her Movie

 Her (stylized in lowercase) is a 2013 American science-fiction romantic drama film written, directed, and co-produced by Spike Jonze. It marks Jonze's solo screenwriting debut. The film follows Theodore Twombly (Joaquin Phoenix), a man who develops a relationship with Samantha (Scarlett Johansson), an artificially intelligent virtual assistant personified through a female voice. The film also stars Amy Adams, Rooney Mara, Olivia Wilde, and Chris Pratt.

 {% youtube "https://www.youtube.com/watch?v=dJTU48_yghs" %}

 More at:
  * [https://en.wikipedia.org/wiki/Her_(film)](https://en.wikipedia.org/wiki/Her_(film))

 See also [H], [AI Movie]
 

# Heuristic

 A heuristic is a mental shortcut that allows an individual to make a decision, pass judgment, or solve a problem quickly and with minimal mental effort. While heuristics can reduce the burden of decision-making and free up limited cognitive resources, they can also be costly when they lead individuals to miss critical information or act on unjust [biases][Bias].

 The study of heuristics was developed by renowned psychologists Daniel Kahneman and Amos Tversky. Starting in the 1970s, Kahneman and Tversky identified several different kinds of heuristics, most notably the availability heuristic and the anchoring heuristic. Since then, researchers have continued their work and identified many different kinds of heuristics, including:
  * Familiarity heuristic
  * Fundamental attribution error
  * Representativeness heuristic
  * Satisficing

  ```
What is the anchoring heuristic?
The anchoring heuristic, or anchoring bias, occurs when someone relies more heavily on the first piece of information learned when making a choice, even if it's not the most relevant. In such cases, anchoring is likely to steer individuals wrong.

What is the availability heuristic?
The availability heuristic describes the mental shortcut in which someone estimates whether something is likely to occur based on how readily examples come to mind. People tend to overestimate the probability of plane crashes, homicides, and shark attacks, for instance, because examples of such events are easily remembered.

What is the representativeness heuristic?
People who make use of the representativeness heuristic categorize objects (or other people) based on how similar they are to known entities—assuming someone described as "quiet" is more likely to be a librarian than a politician, for instance.

What is satisficing?
Satisficing is a decision-making strategy in which the first option that satisfies certain criteria is selected, even if other, better options may exist.

What is the fundamental attribution error?
Sometimes called the attribution effect or correspondence bias, the term describes a tendency to attribute others’ behavior primarily to internal factors—like personality or character—while attributing one’s own behavior more to external or situational factors.
 ```
 When are heuristic wrong?
 ```
What is an example of the fundamental attribution error?
If one person steps on the foot of another in a crowded elevator, the victim may attribute it to carelessness. If, on the other hand, they themselves step on another’s foot, they may be more likely to attribute the mistake to being jostled by someone else.

Listen to your gut, but don’t rely on it. Think through major problems methodically—by making a list of pros and cons, for instance, or consulting with people you trust. Make extra time to think through tasks where snap decisions could cause significant problems, such as catching an important flight.
 ```

 More at:
  * [https://www.psychologytoday.com/us/basics/heuristics](https://www.psychologytoday.com/us/basics/heuristics)
  * [https://www.investopedia.com/terms/h/heuristics.asp](https://www.investopedia.com/terms/h/heuristics.asp)

 See also [H], [Deep Blue Challenge]


# Hidden Layer

 See also [H], [Artificial Neural Network], [Dropout Layer]


# Hidden Markov Model

# HMM

 More at:
  * [https://en.wikipedia.org/wiki/Hidden_Markov_model ](https://en.wikipedia.org/wiki/Hidden_Markov_model )

 See also [H], [Generative Classified], [Hidden State]


# Hidden State

  * Memory from previous RNN stages = representation of previous input
  * In Encoder-decoder = latent state/space?

 See also [H], [H], [Encoder-Decoder Model], [Hidden Markov Model], [Recurrent Neural Network]


# Hinge Loss Function

 The use of hinge loss is very common in binary classification problems where we want to separate a group of data points from those from another group. It also leads to a powerful machine learning algorithm called [Support Vector Machines (SVMs)] Let’s have a look at the mathematical definition of this function.

 More at:
  * [https://www.baeldung.com/cs/hinge-loss-vs-logistic-loss](https://www.baeldung.com/cs/hinge-loss-vs-logistic-loss)

 See also [H], [Loss Function]


# Holdout Fold

 See also [H], [Cross-Validation Sampling Method]


# Holistic Evaluation of Language Model Benchmark

# HELM Benchmark

 A language model takes in text and produces text. Despite their simplicity, language models are increasingly functioning as the foundation for almost all language technologies from question answering to summarization. But their immense capabilities and risks are not well understood. Holistic Evaluation of Language Models (HELM) is a living benchmark that aims to improve the transparency of language models.

 More at:
  * [https://crfm.stanford.edu/helm/latest/](https://crfm.stanford.edu/helm/latest/)
  * [https://github.com/stanford-crfm/helm](https://github.com/stanford-crfm/helm)
  * [https://crfm.stanford.edu/2022/11/17/helm.html](https://crfm.stanford.edu/2022/11/17/helm.html)


 See also [H], ...


# Huber Loss Function

 Now we know that the Mean Square Error (MSE) is great for learning outliers while the Mean Absolute Error (MAE) is great for ignoring them. But what about something in the middle? Consider an example where we have a dataset of 100 values we would like our model to be trained to predict. Out of all that data, 25% of the expected values are 5 while the other 75% are 10. An MSE loss wouldn’t quite do the trick, since we don’t really have “outliers”; 25% is by no means a small fraction. On the other hand we don’t necessarily want to weight that 25% too low with an MAE. Those values of 5 aren’t close to the median (10 — since 75% of the points have a value of 10), but they’re also not really outliers. Our solution? The Huber Loss Function. The Huber Loss offers the best of both worlds by balancing the MSE and MAE together. We can define it using the following piecewise function:

 ![]( {{site.assets}}/h/huber_loss_function_formula.png ){: width="100%"}

 ![]( {{site.assets}}/h/huber_loss_function_graph.png ){: width="100%"}

 What this equation essentially says is: for loss values less than delta, use the MSE; for loss values greater than delta, use the MAE. This effectively combines the best of both worlds from the two loss functions! Using the MAE for larger loss values mitigates the weight that we put on outliers so that we still get a well-rounded model. At the same time we use the MSE for the smaller loss values to maintain a quadratic function near the centre. This has the effect of magnifying the loss values as long as they are greater than 1. Once the loss for those data points dips below 1, the quadratic function down-weights them to focus the training on the higher-error data points.

 ```
import numpy as np

def huber_loss(y_pred, y, delta=1.0):
    huber_mse = 0.5*(y-y_pred)**2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae)
 ```

 Pros and Cons:
  * Advantages : Best of the MSE and the MAE ?
  * Disadvantages
   * For cases where outliers are very important to you, use the MSE! 
   * For cases where you don’t care at all about the outliers, use the MAE!

 See also [H], [Loss Function], [Mean Absolute Error Loss Function], [Mean Square Error Loss Function]


# Human-Centered AI

  * Understand human intelligence more deeply and more broadly
  * Connect to neuroscience, cognitive psychology, etc
  * Collaborates with humans
  * Enhances, not replaces humans; gives humans appropriate control
  * Aware of human preferences (value discovery/alignment)
  * Aware of human abilities and limitations
  * Accountable, explainable, understandable, and trustworthy
  * Focused on what is good for humanity (health, environment)
  * Bridges to policy world, other academic disciplines, industry
  * Respects ethics (animal-centered AI? Earth-centered AI?) 

 More at:
  * stanford HAI - [https://hai.stanford.edu/](https://hai.stanford.edu/)

 See also [H], [Artificial Intelligence]


# Humanoid

 A robot that is meant to resemble a human.

 See also [H], [Robot]


# Hyperparameter

 * Parameters not directly learned by learning algorithm
 * specified outside of training procedure
 * control the capacity of the model i.e. flexibility of model to fit the data
 * prevent overfittting
 * improve the convergence of the gradient descent (training time)

 `~ parameters to tune the performance of the ML model`. Any decision the algorithm author can't make for you. In machine learning, we use the term hyperparameter to distinguish from standard model parameters. So, it is worth to first understand what those are. A machine learning model is the definition of a mathematical formula with a number of parameters that need to be learned from the data. That is the crux of machine learning: fitting a model to the data. This is done through a process known as model training. In other words, by training a model with existing data, we are able to fit the model parameters. `However, there is another kind of parameters that cannot be directly learned from the regular training process`. These parameters express “higher-level” properties of the model such as its complexity or how fast it should learn. They are called hyperparameters. Hyperparameters are usually fixed before the actual training process begins. So, how are hyperparameters decided? That is probably beyond the scope of this question, but suffice to say that, broadly speaking, this is done by setting different values for those hyperparameters, training different models, and deciding which ones work best by testing them.

 So, to summarize. Hyperparameters:
  * Define higher level concepts about the model such as complexity, or capacity to learn.
  * Cannot be learned directly from the data in the standard model training process and need to be predefined.
  * Can be decided by setting different values, training different models, and choosing the values that test better

 Some examples of hyperparameters:
  * Number of leaves or depth of a tree
  * Number of trees
  * Number of latent factors in a matrix factorization
  * Learning rate (in many models)
  * Number of hidden layers in a deep neural network
  * Number of hidden nodes in network layers
  * Number of clusters in a k-means clustering
  * [Drop out function]
  * Regularization
  * Boosting step size
  * Initialization of clustering algorithm

 Beware often you have 1 hyperparameter that is more impactful than the other. Also beware of correlation between parameters. Hyperparameters can take a continuous, integer, or categorical value (ex learning rate 0.1, epochs:20, optimizer: sgd). 

 See also [H], [Algorithmic], [Boosting Step Size], [Complexity], [Data Handling], [Drop Out], [Expressiveness], [Hyperparameter Optimization], [Learning Rate], [Parameter], [Regularization], [XGBoost]


# Hyperparameter Optimization

# HPO

 Process used to tune the [hyperparameters][Hyperparameter] to get the best prediction (best is defined by a function!)

 Tuning strategies:
  1. Trial and Error, defaults, guess, experience, intuition, heuristics 
  1. Try everything using one of the 3 most popular HPO techniques:
   * [Random search] or the derived [Sobol Search]
   * [Grid search]
   * [Bayes Search]
  1. Meta model ... Required to avoid over-fitting and under-fitting.

 :warning: High dimensional grid search, = the [curse of dimensionality]

 In general, if the number of combinations is limited enough, we can use the [Grid Search] technique. But when the number of combinations increases, we should try [Random Search] or [Bayes Search] as they are less computationally expensive.

 See also [H], [AutoML], [CPU], [F1 Score], [GPU], [Hyperparameter], [Meta Model], [Overfitting], [Underfitting]


# Hyperparameter Tuning

 See [Hyperparameter Optimization]


# Hyperplane

 The boundary between TWO classification classes? Yes, in a real or latent dimension! For example in a 3D space, a 2-D plane could be an hyperplane where on one side you have the elements of class A and on the other side you have the elements of class B. Used as a decision boundary. A hyperplane is a subspace that has one dimension less than the ambient space that contains it. In simple linear regression, there is one dimension for the response variable and another dimension for the explanatory variable, for a total of two dimensions. The regression hyperplane thus has one dimension; a hyperplane with one dimension is a line. In mathematics, a hyperplane H is a linear subspace of a vector space V such that the basis of H has cardinality one less than the cardinality of the basis for V.  In other words, if V is an n-dimensional vector space than H is an (n-1)-dimensional subspace.  Examples of hyperplanes in 2 dimensions are any straight line through the origin. In 3 dimensions, any plane containing the origin.  In higher dimensions, it is useful to think of a hyperplane as member of an affine family of (n-1)-dimensional subspaces (affine spaces look and behavior very similar to linear spaces but they are not required to contain the origin), such that the entire space is partitioned into these affine subspaces. This family will be stacked along the unique vector (up to sign) that is perpendicular to the original hyperplane.  This "visualization" allows one to easily understand that a hyperplane always divides the parent vector space into two regions.

 ![]( {{site.assets}}/h/hyperplane.png ){: width="100%"}

 See also [H], [Classification], [Decision Boundary], [Support Vector Machine]
