---
title: Q
permalink: /q/

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


# Quadratic Regression

 See also [Q], [Polynomial Regression], [Regression]


# Quantization Error

 Quantization error is the difference between the analog signal and the closest available digital value at each sampling instant from A/D converter. Quantization error also introduces noise,to the sample signal. Relations The higher the resolution of A/D converter, the lower the quantization error and the smaller the quantization noise.


# Quantized Signal

 ![]( {{site.assets}}/q/quantized_signal.png ){: width="100%"}

 See also [Q], [Quantizer]

# Quantizer

 ![]( {{site.assets}}/q/quantizer.png ){: width="100%"}

 See also [Q], [Quantized Signal]


# Question Answering

# QA

 The Stanford Question Answering Dataset (SQuAD) v1.1 is a collection of 100k crowdsourced question/answer pairs. Given a question and a passage from Wikipedia containing the answer, the task is to predict that answer text span in the passage. SQuAD v2.0 extends SQuAD v1.1 prolem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.
 
 ```
from transformers import BertForQuestionAnswering, AutoTokenizer

modelname = 'deepset/bert-base-cased-squad2'

model = BertForQuestionAnswering.from_pretrained(modelname)      # Transferred learning ? Yes, possibly
tokenizer = AutoTokenizer.from_pretrained(modelname)             # Transferred learning ? No, just import!

from transformers import pipeline
ask_question = pipeline('question-answering', model=model, tokenizer=tokenizer)

context = "The Intergovernmental Panel on Climate Change (IPCC) is a scientific intergovernmental body under the auspices of the United Nations, set up at the request of member governments. It was first established in 1988 by two United Nations organizations, the World Meteorological Organization (WMO) and the United Nations Environment Programme (UNEP), and later endorsed by the United Nations General Assembly through Resolution 43/53. Membership of the IPCC is open to all members of the WMO and UNEP. The IPCC produces reports that support the United Nations Framework Convention on Climate Change (UNFCCC), which is the main international treaty on climate change. The ultimate objective of the UNFCCC is to \"stabilize greenhouse gas concentrations in the atmosphere at a level that would prevent dangerous anthropogenic [i.e., human-induced] interference with the climate system\". IPCC reports cover \"the scientific, technical and socio-economic information relevant to understanding the scientific basis of risk of human-induced climate change, its potential impacts and options for adaptation and mitigation.\""

answer = ask_question({
    'question': 'What organization is the IPCC a part of?',
    'context': context
})


# RETURNS 'answer' as a JSON OBJECT
{'score': 0.4881587028503418,
 'start': 118,
 'end': 132,
 'answer': 'United Nations'}
 ```

  * Extractive Q&A = find the answer in the context (context is passed in the prompt!)
  * Abstractive Q&A = find the answer inside or outside of the context

 See also [Q], [BERT Model], [GPT Model], [Natural Language Processing]


# Question Answering Graph Neural Network

# QAGNN

 A QA LLM NLP used to generate a graph which is then merged with a Knowledge Graph.... to finally answer the question. 
 :warning: LLM is one of the best method to extract entities from text.

 {% pdf "{{site.assets}}/q/qagnn_paper.pdf" %}

 More at:
  * [https://ai.stanford.edu/blog/qagnn/](https://ai.stanford.edu/blog/qagnn/)

 See also [Q], [Entity Extraction], [Graph Neural Network], [Knowledge Graph], [Machine Reasoning], [Question Answering]


# Quora Company

 A Q&A internet company that is jumping in the AI race with the poe interface.

 Quora is a question-and-answer platform where users can ask questions and get answers from a community of users. It was founded in 2009 by two former Facebook employees and has since grown to become one of the largest question-and-answer platforms on the internet. Users can ask any question they have on any topic, and other users can answer the question, provide comments, and upvote or downvote answers. Quora also allows users to follow topics and other users, which can help them discover new questions and answers that are relevant to their interests. Quora is known for its high-quality answers, which are often written by experts in their respective fields.

 More at:
  * announcement - [https://quorablog.quora.com/Poe-1](https://quorablog.quora.com/Poe-1)
  * POE web UI - [https://poe.com/](https://poe.com/)

 See also [Q], ...
