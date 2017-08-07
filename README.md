# Self-Driving Car Program Note

This note is mainly used for record some personal understanding about self-driving car program. 

## Kalman Filter

Kalman Filters are very popular in localization problems. They represents possible location (distributions) by Gaussians and iterates on two main cycles: Measurement Update Cycle and Motion Update/Prediction Cycle. 

In general, the measurement update cycle is believed to connected with Bayes rule. This is true. But, it would be even better if we just consider the 'posterior' as a product of two Gaussion ditributions 'prior' and 'measurement_distribution'. Because the later idea can be more easily promoted. For example, to localize, we possibly can have three or more different measurements methods. Each of them can give us a Gaussian distribution. Then in most of cases, their product have the highest accuracy. 

In self-driving car program, the measurement is from radar or lidar, which should be independent with the prior, or the result of prediction cycle. So to get a better result (posterior), one can always use Bayes rule and do production. 


