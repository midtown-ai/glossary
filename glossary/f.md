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

 Harmonic mean of precision and recall. A metric used for [model evaluation]  in scenarios where both false positives and false negatives are crucial. For instance, in information retrieval or sumarization tasks.

 A Measure of accuracy of a model. Used to find [hyperparameter optimization].

 When to use? F1-Score is used when the False Negatives and False Positives are important. F1-Score is a better metric for Imbalanced Data.

 More at:
  * [https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5](https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5)

 See also [F], [Confusion Matrix], [Hyperparameter Optimization]


# Face Detection

 See also [F], ...


# Facebook Company

 See [Meta Company]


# Fair AI

 More at:
  * [https://mostly.ai/blog/we-want-fair-ai-algorithms-but-how-to-define-fairness](https://mostly.ai/blog/we-want-fair-ai-algorithms-but-how-to-define-fairness)
  * Fair synthetic data generation - [https://mostly.ai/blog/diving-deep-into-fair-synthetic-data-generation-fairness-series-part-5](https://mostly.ai/blog/diving-deep-into-fair-synthetic-data-generation-fairness-series-part-5)

 See also [F], ...


# FAIRSEQ Toolkit

 Built by [Meta] on the top of [PyTorch]

 Includes
  * [Wav2Letter Model]
  * [Wav2Vec Model]
  * ...

 {% youtube "https://www.youtube.com/watch?v=t6JjlNVuBUQ" %}

 More at:
  * code - [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)


# Fake Art

 More at:
  * [https://nypost.com/2023/04/05/how-frightening-new-ai-midjourney-creates-realistic-fake-art/](https://nypost.com/2023/04/05/how-frightening-new-ai-midjourney-creates-realistic-fake-art/)

 See also [F], ...


# Fake News

  More at:
   * SIFT method - [https://oer.pressbooks.pub/collegeresearch/chapter/the-sift-method/](https://oer.pressbooks.pub/collegeresearch/chapter/the-sift-method/)

  See also [F], ...


# Falcon Model

 ```
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" # You can also use the larger model falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=10000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

from langchain import PromptTemplate,  LLMChain

template = """
You are an ethical hacker and programmer. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Create a python script to send a DNS packet using scapy with a secret payload "

print(llm_chain.run(question))

 ```

 More at:
  * colab - [https://colab.research.google.com/drive/1rLShukC14BodnSI9OTyBLu9v0CVrrCsi?usp=sharing](https://colab.research.google.com/drive/1rLShukC14BodnSI9OTyBLu9v0CVrrCsi?usp=sharing)

 See also [F], ...


# Fashion MNIST Dataset

 See also [F], [MNIST Dataset]


# Fast Random Projection

# FastRP

 FastRP, a scalable and performant algorithm for learning distributed node representations in a graph. FastRP is over 4,000 times faster than state-of-the-art methods such as [DeepWalk] and [node2vec], while achieving comparable or even better performance as evaluated on several real-world networks on various downstream tasks. We observe that most network embedding methods consist of two components: construct a node similarity matrix and then apply dimension reduction techniques to this matrix. 

 {% youtube "https://www.youtube.com/watch?v=uYvniQlSvyQ" %}

 {% pdf "https://arxiv.org/pdf/1908.11512.pdf" %}

 More at:
  * paper - [https://arxiv.org/abs/1908.11512](https://arxiv.org/abs/1908.11512)
  * articles
    * [https://towardsdatascience.com/behind-the-scenes-on-the-fast-random-projection-algorithm-for-generating-graph-embeddings-efb1db0895](https://towardsdatascience.com/behind-the-scenes-on-the-fast-random-projection-algorithm-for-generating-graph-embeddings-efb1db0895)

 See also [F], ...


# Feature

 :warning: features are expected to be independent variables between each other (and the output/prediction is dependent on the inputs/features)

 `Input parameter to the model`. Which features are important to make the right prediction? Beware that the dataset needs to be at least 10 times the number of features.

 See also [F], [Data Point], [Dataset], [Explanatory Variable], [Feature Engineering], [Feature Extraction], [Feature Vector], [Naive Bayes]


# Feature Attribution

 Feature Attributions is a family of methods for explaining a model’s predictions on a given input by attributing it to features of the individual inputs. The attributions are proportional to the contribution of the feature to the prediction. They are typically signed, indicating whether a feature helps push the prediction up or down. Finally, attributions across all features are required to add up to the model’s prediction score.

 Feature Attributions have been successfully used in the industry and also at Google to improve model transparency, debug models, and assess model robustness. Prominent algorithms for computing feature attributions include SHAP, Integrated Gradients and LIME. Each algorithm offers a slightly different set of properties.

 More at:
  * [https://cloud.google.com/blog/topics/developers-practitioners/monitoring-feature-attributions-how-google-saved-one-largest-ml-services-trouble](https://cloud.google.com/blog/topics/developers-practitioners/monitoring-feature-attributions-how-google-saved-one-largest-ml-services-trouble)

# Feature Distribution

 See also [F], ...


# Feature Distribution Transformation

 Transformations:
  * [Box Cox Transformation]
  * [Log Transformation]

 See also [F], ...


# Feature Engineering

 ~ a step in a [machine learning pipeline]

 Feature engineering is an iterative process that often requires experimentation and creating many models to find the input features that give the best model performance. You want features that are weakly correlated with each other, but that strongly correlated with the output. 

 Examples:
  * Missing data imputation
  * Variable discretization
  * Handling [outliers][Outlier]
  * Creating features from dates and time
  * Extracting features from relational data and time series
  * Extracting features from text
  * Numeric features may need to be [rescaled][Feature Scaling]
  * The values of [categorical features][Categorical Feature] may need to be [encoded][Encoding] (Monday=1, ..., Sunday =7, or one hot encoding? <!> In first encoding, you pass an incorrect hierachical information!)
  * Features may need to be parsed into multiple fields
  * Techniques like [Principal Component Analysis (PCA)][Principal Component Analysis] may need to be applied to extract new features
  * Features may need to [reshaped][Feature Distribution Transformation] to conform to statistical distribution, such as normal/Gaussian.

 Beware that the data-set needs to be at least 10 times the number of features. Example: for call routing in a call center
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


# Feature Importance

 Feature importance is a technique used in [machine learning] to determine the relative importance of each input feature or predictor variable in predicting the target variable. It allows us to identify which features are most relevant or informative for making accurate predictions.

 In many machine learning models, including [decision trees], [random forests], and [gradient boosting], feature importance can be calculated based on how much each feature reduces the uncertainty or error of the model when it is used to make predictions. The most important features are those that lead to the greatest reduction in uncertainty or error.

 Feature importance can be used for a variety of purposes, such as identifying which features to focus on when collecting new data, identifying potential problems with the model, and explaining how the model is making its predictions. It is also useful for [feature selection], which involves choosing a subset of the most important features to include in the model, in order to improve its [accuracy] and reduce [overfitting].

 Problems solved by feature importance:
  * [Data Leakage] check
  * ...

 ```
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier object
dt = DecisionTreeClassifier(random_state=42)

# Fit the model on training data
dt.fit(X_train, y_train)

# Calculate feature importance
importance = dt.feature_importances_

# Print feature importance scores
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (iris.feature_names[i], v))

# Make predictions on testing data
y_pred = dt.predict(X_test)

# Evaluate the model performance on testing data
accuracy = dt.score(X_test, y_test)
print("Accuracy:", accuracy)
 ```

 See also [F], ...


# Feature Learning

 before classifier training!

 In machine learning, [feature] learning or representation learning is a set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task.

 Feature learning is motivated by the fact that machine learning tasks such as classification often require input that is mathematically and computationally convenient to process. However, real-world data such as images, video, and sensor data has not yielded to attempts to algorithmically define specific features. An alternative is to discover such features or representations through examination, without relying on explicit algorithms.

 Feature learning can be either [supervised], [unsupervised] or [self-supervised].

 More at:
  * [https://en.wikipedia.org/wiki/Feature_learning](https://en.wikipedia.org/wiki/Feature_learning)

 See also [F], ...


# Feature Normalization

 Cleaning the data in preparation of feeding it to a model.

 Transform [features][Feature] to an explicit range between 0 and 1 for example.

 ```
           X - Xmin
Xnorm = --------------
          Xmax - Xmin
 ```

 See also [F], [Feature Scaling]


# Feature Scaling

 Methods:
  * [Feature Normalization]
  * [Feature Standardization]

 See also [F], ...


# Feature Selection

 ~ a step in the [machine learning pipeline]

 See also [F], ...


# Feature Standardization

 Transform [features][Feature] to a measure of how each value differs from the mean.

 The new value range typically from -3 to 3 (+- 3 standard-deviation)

 ```
          X - mean(X)
Xnorm = ---------------
           stddev(X)
 ```

 See also [F], [Feature Scaling]


# Feature Vector

 Single column matrix (a vector) that contains all the inputs to a model (ex: artificial neuron). 

 See also [F], [Feature], [Vector]


# Feature Visualization

 There is a growing sense that neural networks need to be interpretable to humans. The field of neural network interpretability has formed in response to these concerns. As it matures, two major threads of research have begun to coalesce: feature visualization and attribution.

 {% youtube "https://www.youtube.com/watch?v=McgxRxi2Jqo" %}

 Convolutional networks

 {% pdf "https://arxiv.org/pdf/1311.2901.pdf" %}

 More at:
  * [https://yosinski.com/deepvis](https://yosinski.com/deepvis)
  * [https://sander.ai/2014/08/05/spotify-cnns.html](https://sander.ai/2014/08/05/spotify-cnns.html)
  * [https://github.com/yosinski/deep-visualization-toolbox](https://github.com/yosinski/deep-visualization-toolbox)
  * [https://distill.pub/2017/feature-visualization/](https://distill.pub/2017/feature-visualization/)
  * papers
   * Visualizing CNN - [https://arxiv.org/abs/1311.2901](https://arxiv.org/abs/1311.2901)

 See also [F], ...


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


# Fei-Fei Li Person

 * Studied at caltech
 * Launched the image project [ImageNet] in 2007
 * Led the computer vision lab at Stanford

 {% youtube "https://www.youtube.com/watch?v=40riCqvRoMs" %}

 {% youtube "https://www.youtube.com/watch?v=lV29vATjoW4" %}

 See also [F], [People]


# Few-Shot Learning

 A [prompt engineering] technique for [large language models][LLM]!

 In addition to the task description the model sees a few examples of the task. No gradient updates are performed.

 ```
Translate English to French                # Task description
sea otter => loutre de mer                 # Example 1
peppermint => menthe poivree               # ...
plush girafe => girafe peluche             # Example N
cheese =>
 ```

 Deductions from a few hand-picked examples. ex: You watch someone playing a game. After he or she  played 5 round,, you say, oh yes, I think I can play the game. Few-, one-, and zero-shot settings are specialized cases of zero-shot task transfer. In a few-shot setting, the model is provided with a task description and as many examples as fit into the context window of the model. In a one-shot setting, the model is provided with exactly one example and, in a zero-shot setting, with no example. 

 ![]( {{site.assets}}/f/few_shot_learning_accuracy.png ){: width="100%"}

 {% pdf "https://arxiv.org/pdf/2005.14165.pdf" %}

 More at:
  * [https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing)](https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing))
  * paper - 

 See also [F], [One-Shot Learning], [Zero-Shot Task Transfer], [Zero-Shot Learning], 


# Few-Shot Prompting

 ~ [Few-shot learning] applied on [Prompt Engineering]

 See also [F], ...


# Few-Shot Reinforcement Learning
# Few-Shot RL

 Learn new tasks from only a few examples. Leverages prior knowledge.

 More at:
  * ...

 See also [F], ...

# File Mode

 Each step of the training process generate file and are not streamed (?). 

 See also [F], [Pipe Mode]


# Fine-Tuning

 More at:
   * GPT fine-tuning - [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

 See [Supervised Fine-Tuning]


# FinGPT Model

 A model developed by the [AI Finance Foundation]

 {% pdf "https://arxiv.org/pdf/2306.06031.pdf" %}

 {% youtube "https://www.youtube.com/watch?v=CH3BdIvWxrA" %}

 More at:
  * AI finance foundation - [https://github.com/AI4Finance-Foundation](https://github.com/AI4Finance-Foundation)
  * paper - [https://arxiv.org/abs/2306.06031](https://arxiv.org/abs/2306.06031)
  * code - [https://github.com/AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
  * articles
    * [https://medium.datadriveninvestor.com/fingpt-powering-the-future-of-finance-with-20-cutting-edge-applications-7c4d082ad3d8](https://medium.datadriveninvestor.com/fingpt-powering-the-future-of-finance-with-20-cutting-edge-applications-7c4d082ad3d8)

  See also [F], [IndexGPT Model], [LLaMa Model]


# Flamingo Model

 A [visual language model] developed at [DeepMind]

 {% youtube "https://www.youtube.com/watch?v=yM9S3fLp5Gc" %}

 {% pdf "{{site.assets}}/f/flamingo_model_paper.pdf" %}

 See also [F], [IDEFICS Model]


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


# Formal Reasoning

 Formal [reasoning] is a systematic and logical process that follows a set of rules and principles. It is characterized by its structured and rigorous approach, often used in disciplines like mathematics, formal logic, and computer science. Formal reasoning relies on deductive logic and mathematical proofs to arrive at valid conclusions. It involves applying established rules and principles to solve problems and make deductions.

 More at:
  * LLM reasoning ability - [https://www.kaggle.com/code/flaussy/large-language-models-reasoning-ability](https://www.kaggle.com/code/flaussy/large-language-models-reasoning-ability)

 See also [F], ...


# Foundational Model

 A foundational model is a basic machine learning model that is built from scratch using a set of rules and parameters, without any pre-existing knowledge or training data. It requires manual tuning and optimization to perform well on specific tasks.

 On the other hand, a [pre-trained model] is a [machine learning] model that has already been trained on a large amount of data and optimized for a specific task. Pre-training typically involves training a model on a large dataset, often using unsupervised learning techniques, to learn general features and patterns in the data. The pre-trained model can then be fine-tuned on a smaller, task-specific dataset to improve its performance on a particular task.

 The main difference between a foundational model and a pre-trained model is the level of training and optimization required. Foundational models require extensive manual tuning and optimization to perform well on a specific task, while pre-trained models have already undergone extensive training and optimization, and can be fine-tuned for specific tasks with relatively little additional training.

 Pre-trained models have become increasingly popular in recent years due to their ability to quickly achieve state-of-the-art performance on a wide range of tasks, without requiring extensive manual tuning and optimization.

 More at:
  * [https://en.wikipedia.org/wiki/Foundation_models](https://en.wikipedia.org/wiki/Foundation_models)

 See also [F], ...


# Frequent Pattern Growth Algorithm

# FP-Growth Algorithm

 ~ a type of [unsupervised learning] that is used for [association rule]

 The FP-growth (Frequent Pattern growth) algorithm is an efficient method for mining frequent itemsets and generating association rules without candidate generation. Here are the key points about FP-growth:

  * It uses a divide-and-conquer approach to compress a large database into a compact FP-tree (frequent pattern tree) representation.
  * The FP-tree stores quantified itemset information in a compressed form and avoids costly database scans.
  * It decomposes the mining task into smaller tasks in a recursive fashion by partitioning the database on frequent itemsets.
  * Each partition is represented by a conditional FP-tree which is used to recursively grow frequent patterns.
  * It uses a pattern-fragment growth method to avoid the costly generation of candidate sets.
  * It uses a pattern growth approach instead of the candidate generation and test approach of Apriori-like algorithms.
  * Performance is improved by orders of magnitude compared to Apriori because of the compressed representation and no candidate generation.
  * Works well for mining long patterns and in dense databases.

 In summary, the FP-growth algorithm mines frequent itemsets by recursively building conditional FP-trees and joining frequent itemset fragments. This divide-and-conquer approach avoids costly database scans and expensive candidate generation.

 More at:
  * ...

 See also [F], ...


# Fully Connected Layer
# FC Layer

 A List of feature values becomes a list of votes (which are weighted to following layers).

 ![]( {{site.assets}}/f/fully_connected_layer.png ){: width="100%}

 See also [F], [Convolutional Layer], [Convolutional Neural Network], [Polling Layer], [ReLU Layer]


# Function Estimation

 Here we are trying to predict a variable y given an input vector x. We assume that there is a function f(x) that describes the approximate relationship between y and x. For example, we may assume that y = f(x) + ε, where ε stands for the part of y that is not predictable from x. In function estimation, we are interested in approximating f with a model or estimate fˆ. Function estimation is really just the same as estimating a parameter θ; the function estimator fˆis simply a point estimator in function space. Ex: in polynomial regression we are either estimating a parameter w or estimating a function mapping from x to y. 

 See also [F], [Estimator], [Point Estimator]


# Fused Kernel

 Fused kernels, in the context of deep learning, refer to a technique that combines multiple computational operations into a single kernel or operation. The purpose of fusing kernels is to improve computational efficiency and reduce memory overhead by minimizing data movement and kernel launch overhead.

 In deep learning models, there are often multiple operations performed on the same set of data. These operations can include element-wise operations, matrix multiplications, convolutions, and more. Fusing these operations means combining them into a single operation that performs all the required computations simultaneously, reducing the need for intermediate storage and data transfer between different kernel launches.

 By fusing kernels, the computational efficiency can be improved in several ways:

 Reduced memory overhead: Fusing kernels eliminates the need to store intermediate results in memory, resulting in lower memory usage. This is particularly beneficial when dealing with large tensors or when memory resources are limited.

 Minimized data movement: Fused kernels perform multiple operations on the same data without the need for transferring data between different kernel launches. This reduces the data movement overhead, improving performance.

 Enhanced hardware utilization: Fusing kernels allows for better utilization of hardware resources, such as [GPU] cores. By executing multiple operations in a single kernel, the GPU cores can be fully utilized, leading to improved parallelism and faster computation.

 Fused kernels are commonly used in deep learning frameworks and libraries to optimize the execution of neural network models. They are implemented through specialized libraries or compiler optimizations that identify opportunities for fusion and generate efficient code that combines multiple operations into a single kernel. The specific techniques and mechanisms for kernel fusion may vary depending on the deep learning framework or library being used.

 More at:
  * [https://www.surfactants.net/creating-a-fused-kernel-in-pytorch/](https://www.surfactants.net/creating-a-fused-kernel-in-pytorch/)
  * [https://stackoverflow.com/questions/56601075/what-is-a-fused-kernel-or-fused-layer-in-deep-learning](https://stackoverflow.com/questions/56601075/what-is-a-fused-kernel-or-fused-layer-in-deep-learning)
  * [https://towardsdatascience.com/how-to-increase-training-performance-through-memory-optimization-1000d30351c8](https://towardsdatascience.com/how-to-increase-training-performance-through-memory-optimization-1000d30351c8)

 See also [F], [Activation Checkpointing], [GPU], [Zero Redundancy Optimization]


# Futuri Media Company

 Find out how we help you grow your content, audience, and revenue.

 Products:
  * [RadioGPT]
  * TopicPulse: Story analysis to see the real-time time evolution of a topic
  * FuturiStreaming: Get stats on streams, number of listener, etc

 More at:
  * [https://futurimedia.com/](https://futurimedia.com/)

 See also [F], ...
