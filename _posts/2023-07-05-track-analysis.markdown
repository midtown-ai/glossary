---
layout: post
# layout: single
title:  "DeepRacer Track Analysis"
date:   2023-07-05 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}

* toc
{:toc}


## Links

 * optimal racing line
   * ideal racing line - [https://blog.orium.com/the-best-path-a-deepracer-can-learn-2a468a3f6d64](https://blog.orium.com/the-best-path-a-deepracer-can-learn-2a468a3f6d64)
   * racing line Bayesian optimization paper - [https://arxiv.org/abs/2002.04794](https://arxiv.org/abs/2002.04794)
   * paper - [https://dspace.mit.edu/handle/1721.1/64669](https://dspace.mit.edu/handle/1721.1/64669)
   * paper - [https://www.remi-coulom.fr/Thesis/](https://www.remi-coulom.fr/Thesis/)
   * racing line article - [https://medium.com/adventures-in-autonomous-vehicles/genetic-programming-for-racing-line-optimization-part-1-e563c606e502](https://medium.com/adventures-in-autonomous-vehicles/genetic-programming-for-racing-line-optimization-part-1-e563c606e502)
   * code
     * compute using PSO - [https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO](https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO)
     * code - [https://github.com/cdthompson/deepracer-k1999-race-lines](https://github.com/cdthompson/deepracer-k1999-race-lines)

 * optimal action space
   * [https://www.linkedin.com/pulse/aws-deepracer-how-calculate-best-racing-line-compute-actions-chen/](https://www.linkedin.com/pulse/aws-deepracer-how-calculate-best-racing-line-compute-actions-chen/)

## Terminology



 Braking point = where you start braking while entering the curve

 turning point = Passed the braking point, where you start steering and turning in the curve. You drive towards the APEX of the curve
  * car should be still braking

 APEX = Where the car touch the inside of the curve
  * before APEX, car is braking (and possibly start re-accelerating)
  * after APEX, car is accelerating (or maintaining same speed)
  * :warning: when you start re-accelerating is when you open the steering

 Exit point = where you stop turning? Maximum speed at exit if straight line after?
  * to open the exit, use an APEX after the geometric line (turn more first, to accelerate more later)
    * the turn = 1 big turn + 1 straight line! = less time turning than in geometric line
    * brake a bit later (brake point is later), but turn is slower (and accelerate earlier!)

 Racing Line = The trajectory taken by the car on a section of the track.
  1. braking point
  2. turning point
  3. APEX
  4. exit point

 Geometric line
   * mathematical best line
   * the best use of the circuit
   * constant radius using compass
   * every inch of track-outside-inside-outside
   * IF decrease in radius (eg not using all track) = slow speed
   * Geometric line is the fastest line in the corner is in complete isolation

 Segment of corner
   * breaking, turning, apex, and exit points
   * line depends on what follows
   * depends on your car

 The ideal racing line
   * straights are critical
   * APEX later, open up exit
   + less time in corner, more on straight
   + accelerate earlier, full throttle sooner
   + later turning = later braking
   - later turning = higher turn

 The radius if a corner
   * tighter corner --> more focus on exit
   * due to speed - the faster you are going, the slower you will accelerate (air resistance)
   * hairpin more focused on exit (more acceleration to take place)
   * fast curve = keeping up momentum (less acceleration to have / place )





## Direction space

### Directional change analysis

  negative values = right turns

  the circuit requires more pronounced left than right turning = counter clockwise race !

  ![]( {{site.assets}}/_/d/directional_change_plot.png ){: width="100%"}


##  Optimal action space

 * [https://www.linkedin.com/pulse/aws-deepracer-how-calculate-best-racing-line-compute-actions-chen/](https://www.linkedin.com/pulse/aws-deepracer-how-calculate-best-racing-line-compute-actions-chen/)

## Ideal Racing Line

 In simple terms, the ideal racing line is the fastest possible path through a circuit. There are many well-known guidelines for defining an ideal line, but in general, they aim to balance four (often conflicting) goals:
  * Use the full width of the track, to achieve the widest possible turning radius around a curve, because the wider the turn, the faster you can go through it.
  * Keep braking to a minimum (in terms of both deceleration and time) and try to get it all done before you start turning.
  * Accelerate throughout the curve and, if at all possible, exit the curve at full throttle.
  * Drive in a straight line as much as possible, because that maximizes the time you will be driving at top speed.

 Beyond these broad guidelines, defining an ideal line is more art than science; and racing it effectively, depends on everything from the load balancing of the car, to the nature of the circuit, to the reaction time and state of mind of the driver.

  ![]( {{site.assets}}/_/i/ideal_racing_line.png ){: width="100%"}

  Articles
   * [https://medium.com/adventures-in-autonomous-vehicles/genetic-programming-for-racing-line-optimization-part-1-e563c606e502](https://medium.com/adventures-in-autonomous-vehicles/genetic-programming-for-racing-line-optimization-part-1-e563c606e502)

  {% youtube "https://www.youtube.com/watch?v=aZlOkt1oU2k" %}

  {% youtube "https://www.youtube.com/watch?v=bOw9nMbHDIQ" %}

  {% pdf "https://www.paradigmshiftracing.com/uploads/4/8/2/6/48261497/the_perfect_corner_sample.pdf" %}

  {% pdf "https://www.paradigmshiftracing.com/uploads/4/8/2/6/48261497/the_perfect_corner_2_-_sample.pdf" %}

  {% pdf "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-111.pdf" %}

  {% pdf "https://arxiv.org/pdf/2002.04794.pdf" %}

  {% pdf "{{site.assets}}/_/r/racing_lline_optimization_paper.pdf" %}

  {% pdf "https://arxiv.org/pdf/2104.11106.pdf" %}

  {% pdf "https://arxiv.org/pdf/1902.00606.pdf" %}

  {% pdf "https://arxiv.org/pdf/2003.04882.pdf" %}

  {% pdf "https://www.remi-coulom.fr/Publications/Thesis.pdf" %}

## Waypoint visualization

 {% youtube "https://www.youtube.com/watch?v=bv5VqjWId0o" %}
