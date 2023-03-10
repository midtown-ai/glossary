---
title: Z
permalink: /z/

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


# Zero-Shot Learning

  The model predicts the answer given only a natural language description of the task. No gradient updates are performed.
 
 ```
Translate English to French              # Task description
cheese =>                                # Prompt
 ```

 ~ Deductions from examples not seen before. Zero-shot learning is the ability of a model to perform a task without having seen any example of that kind in the past; the model is supposed to understand the task without looking at any examples.  Use near comparables. `when not enough data --> get info from other source in different format, i.e. words for image`. You have seen many cats and dogs, but you have never seen a horse in the data. Pick attributes of ... ex number of legs. (classification based on the attributes) A horse has fur, 4 legs, in a range of colors, .... you are a horse. You need a predefined map of attributes (maybe by reading an encyclopedia) and look for those attributes in the image classifier. Few-, one-, and zero-shot settings are specialized cases of zero-shot task transfer. In a few-shot setting, the model is provided with a task description and as many examples as fit into the context window of the model. In a one-shot setting, the model is provided with exactly one example and, in a zero-shot setting, with no example.

 ![]( {{site.assets}}/z/zero_shot_learning.png ){: width="100%"}

 In Zero-Shot Learning, the data consists of the following:
  * Seen Classes: These are the data classes that have been used to train the deep learning model.
  * Unseen Classes: These are the data classes on which the existing deep model needs to generalize. Data from these classes were not used during training.
  * Auxiliary Information: Since no labeled instances belonging to the unseen classes are available, some auxiliary information is necessary to solve the Zero-Shot Learning problem. Such auxiliary information should contain information about all of the unseen classes, which can be descriptions, semantic information, or word embeddings.

 `Q:` To find a horse using a cat-dog model, do we need to build the model using the semantic attributes for the cat and dog? i.e the semantic attribute that will be used to find the horse (ex: has a tail, fur, color is brown, black, or white, etc)

 See also [Z], [Data Augmentation], [Few-Shot Learning], [Image Classifier], [Insufficient Data Algorithm], [One-Shot Learning], [Semantic Space], [Zero-Shot Task Transfer]


# Zero-Shot Task Transfer

 Zero-shot task transfer is a setting in which the model is presented with few to no examples and asked to understand the task based on the examples and an instruction. Few-, one-, and zero-shot settings are specialized cases of zero-shot task transfer. In a few-shot setting, the model is provided with a task description and as many examples as fit into the context window of the model. In a one-shot setting, the model is provided with exactly one example and, in a zero-shot setting, with no example.

 See also [Z], [Few-shot Learning], [One-Shot Learning], [Zero-Shot Learning]
