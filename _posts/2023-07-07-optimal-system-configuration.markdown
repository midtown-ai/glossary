---
layout: post
# layout: single
title:  "Optimal system configuration"
date:   2023-07-07 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}

* toc
{:toc}


## Links

## Terminology

## Optimal system

 * GD batch size 64
 * entropy 0.01 (default)  <== epsilon !
 * discount factor 0.985 (shorter view than default 0.999)
 * loss function = huber
 * learning rate = 0.0004 (default is 0.0003) so more aggressive!
 * episodes/iteration = 32 (default is 20)
 * number of epoch = 3 (default)
 * 8 workers   <=== how ? DeepRacer for cloud
   * https://www.youtube.com/watch?v=lTg4yCaKEr4
