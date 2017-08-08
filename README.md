# Self-Driving Car Program Note

This note is mainly used for record some personal understanding about self-driving car program. 

## Kalman Filter

Kalman Filter is a popular technique that used in tracking. The idea is to represents the location of tracking object by Gaussian and iterates on two main cycles - Measurement Update Cycle and Motion Update/Prediction Cycle - to obtain a more accurate distribution than those based on a single measurement alone. 

*In general, the measurement update cycle is believed to connected with Bayes rule. This is true. But, it would be better if one can consider the 'posterior' just as the product of two Gaussion ditributions 'prior' and 'measurement_distribution'. In this form, one can easily generalize the ideas to more than three measurement methods. For example, during localization, we possibly have three or more different measurement sources. Each of them can give us a Gaussian distribution. Then in most of cases, their product have the highest accuracy than those based on a single measurement alone. *

In self-driving car program, the measurement is from radar or lidar, which should be independent with the prior, namely, the result of prediction cycle. So to get a better result (posterior), one can always use Bayes rule and do production. 

The **BEAUTY** of Kalman Filter is it can figure out, even though it never directly measure the velocity of the project, and from there is able to make predictions about future locations that incorporate velocity. The mathematical reason behind this is as follows.
```
x' = x + \delta_t * \dot{x}
```
**Explanation** with 1-D situation:

See picture

**Conclusion** - the variables of Kalman Filter

```
                           -          OBSERVABLES
   KALMAN FILTER          |           (e.g. the momentary location)
       STATE              |                 |  |
(e.g. position, speed    <                  V  V
 of the car)              |           HIDDEN
                          |           (e.g. velocity)
                           -
```



