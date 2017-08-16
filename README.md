# Self-Driving Car Engineer Nano-degree Program Note

This note is mainly used to record my personal understandings or observation on self-driving car program. 

**Udacity's Self-driving Car Engineer Nanodegree programme covers machine learning, deep learning, computer vision, sensor fusion, localization, control, and related automotive hardware skills. (NOT ACCURATE and NOT SUFFICIENT)**

## Kalman Filter

Kalman Filter is a popular technique that used in tracking. The idea is to represents the location of tracking object by Gaussian and iterates on two main cycles - Measurement Update Cycle and Motion Update/Prediction Cycle - to obtain a more accurate distribution than those based on a single measurement alone. 

The **BEAUTY** of Kalman Filter is it can figure out, even though it never directly measure the velocity of the project, and from there is able to make predictions about future locations that incorporate velocity. The mathematical reason behind this is as follows.
```
x' = x + \delta_t * \dot{x}
```
**Explanation** with 1-D situation:

See picture

**Conclusion** - the variables of Kalman Filter:

```
                           -          OBSERVABLES
   KALMAN FILTER          |           (e.g. the momentary location)
       STATE              |                 |  |
(e.g. position, speed    <                  V  V
 of the car)              |           HIDDEN
                          |           (e.g. velocity)
                           -
```
**Design of Kalman Filter**:
```
      x' = x + \delta_t * \dot{x}
\dot{x}' = \dot{x}

Let F = [[1, delta_t],[0, 1]], 
    H = [1, 0].

UPDATE
  Predition:
    x' = F*x + u
    P' = F*P*Ft
  Measurement Update:
    y = z - H*x
    S = H*P*Ht + R
    K = P*Ht*Si
    x' = x + K*y
    P' = (I - K*H)*P
  where, x = estimate
         P = uncertainty covariance
         F = state transitivity matrix
         Ft: transpose of F
         u = motion vection
         z = measurement
         H = measurement function
         R = measurement noise
         I = identity matrix
         Si: inverse of S
```
**Test Code**: From Udacity

To observe the function of Kalman Filter, one can revise the value of *measurements*. For example,

  * measurements = [1, 2, 3]
  * measurements = [1, 2, 3, 4, 5]
  * measurements = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15]
  * measurements = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

You will find the accuracy heavily depend on the model you used. Here, in the code, the constant velocity (CV) model was used.

```
# Write a function 'kalman_filter' that implements a multi-
# dimensional Kalman Filter for the example given

from math import *

class matrix:
    
    # implements basic operations of a matrix class
    
    def __init__(self, value):
        self.value = value
        self.dimx = len(value)
        self.dimy = len(value[0])
        if value == [[]]:
            self.dimx = 0
    
    def zero(self, dimx, dimy):
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx = dimx
            self.dimy = dimy
            self.value = [[0 for row in range(dimy)] for col in range(dimx)]
    
    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError, "Invalid size of matrix"
        else:
            self.dimx = dim
            self.dimy = dim
            self.value = [[0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1
    
    def show(self):
        for i in range(self.dimx):
            print self.value[i]
        print ' '
    
    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError, "Matrices must be of equal dimensions to add"
        else:
            # add if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] + other.value[i][j]
            return res
    
    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError, "Matrices must be of equal dimensions to subtract"
        else:
            # subtract if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] - other.value[i][j]
            return res
    
    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:
            raise ValueError, "Matrices must be m*n and n*p to multiply"
        else:
            # subtract if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, other.dimy)
            for i in range(self.dimx):
                for j in range(other.dimy):
                    for k in range(self.dimy):
                        res.value[i][j] += self.value[i][k] * other.value[k][j]
            return res
    
    def transpose(self):
        # compute transpose
        res = matrix([[]])
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res
    
    # Thanks to Ernesto P. Adorio for use of Cholesky and CholeskyInverse functions
    
    def Cholesky(self, ztol=1.0e-5):
        # Computes the upper triangular Cholesky factorization of
        # a positive definite matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        for i in range(self.dimx):
            S = sum([(res.value[k][i])**2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else:
                if d < 0.0:
                    raise ValueError, "Matrix not positive-definite"
                res.value[i][i] = sqrt(d)
            for j in range(i+1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(self.dimx)])
                if abs(S) < ztol:
                    S = 0.0
                res.value[i][j] = (self.value[i][j] - S)/res.value[i][i]
        return res
    
    def CholeskyInverse(self):
        # Computes inverse of matrix given its Cholesky upper Triangular
        # decomposition of matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1, self.dimx)])
            res.value[j][j] = 1.0/tjj**2 - S/tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = -sum([self.value[i][k]*res.value[k][j] for k in range(i+1, self.dimx)])/self.value[i][i]
        return res
    
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res
    
    def __repr__(self):
        return repr(self.value)


########################################

# Implement the filter function below

def kalman_filter(x, P):
    for n in range(len(measurements)):
        
        # measurement update
        z = matrix([[measurements[n]]])
        Y = z - H * x
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse()
        x = x + K * Y
        P = (I - K * H) * P

        # prediction
        x = F * x + u
        P = F * P * F.transpose()
        print("Round {}:\nx = {}\nP = {}\n".format(n+1, x, P))
    return x,P

############################################
### use the code below to test your filter!
############################################

measurements = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
#measurements = [1, 3, 5, 7, 9]

x = matrix([[0.], [0.]]) # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]]) # external motion
F = matrix([[1., 1.], [0, 1.]]) # next state function
H = matrix([[1., 0.]]) # measurement function
R = matrix([[1.]]) # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]]) # identity matrix

print kalman_filter(x, P)
```


## Extended Kalman Filters (EKF)

## Unscented Kalman Filters (UKF)
The Unscented Kalman Filter is an alternative technique to deal with nonlinear process models or nonlinear measurement models. But instead of linearizing a nonlinear function, the UKF uses sigma points to approximate the probability distribution. This idea has at least two advantages. First, UKF allows to use more realistic models, even more complex, to describe the motion of a tracking object. Second, in many cases the sigma points approximate the nonlinear transition better than a linearization does. Some people also consider the unnecessary calculation of Jacobian matrix as the third one.

**Processing Models**

* constant velocity model (CV)
* constant turn rate and velocity magnitude model (CTRV)
* constant turn rate and acceleration (CTRA)
* constant steering angle and velocity (CSAV)
* constant curvature and acceleration (CCA)

```
State vector:    x
Process Model:   x_{k+1} = f(x_k, v_k)
Here, v_k is the process noise vector
```

**One useful skill in deriving of a process model**

Figure out how the change rate of state $\dot{x}$ depends on the state $x$.

```
\dot{x} = \frac{\partial x}{\partial t} = g(x(t)) = ?
```
As a result, we have
```
x_{k+1} = x_k + \int_{t_k}^{t_{k+1}} g(x(t)) dt
```
**Kalman Filter vs Extended Kalman Filter vs Unscented Kalman Filter**

A standard Kalman filter can only handle linear equations. Both the extended Kalman filter and the unscented Kalman filter allow you to use non-linear equations; the difference between EKF and UKF is how they handle non-linear equations.


## Localization

**Aim**: To figure out where is our car in a given map with an accuracy of 10 cm or less.

**Idea**: Let's describe the possible location of our vehicle by a distribution. Then the aim of localization is to improve a high accuracy of this distribution. Similar to tracking problems, localization algorithms use sense (for landmarks) / measurement update and move (prediction) cycles to improve the accuracy of this distribution.
```
    |-----------------------|                  |----------------|
    | Sense                 |   ----------->   |  Move          |
    |                       |      Belief      |                |
    | (Measurement Update)  |   <-----------   |  (Prediction)  |
    |-----------------------|                  |----------------|
```

### Markov Localization

This subsection is one try to solve above problem. How? Use all of relevant history information to give a better estimation of the belief of current position of vehicle. 

For convenience, denote 
```
z_{1:t}    Observation (such as range measurements, bearing, images, ...)
u_{1:t}    Controls (yaw, pitch, roll rates, velocities, ...)
m          Map info

x_t        Current position (2D)

Note. u_t denote the controls at time t-1 that will affect the position of the vehicle at time t; similarly, z_t denote the observations from the vehicle at time t.
```
Then, one can write
```
bel(x_t) = p(x_t | z_{1:t}, u_{1:t}, m)
```

Up to here, we will meet two problems: a) in each cycle of updata and prediction, we need to handle more than 100 GBs data, and b) the amout of data increase over time. If we think about the nature of our former formulation, It is not hard to find out the reason of obtaining a better estimation of the vehicle's location is due to the fact that all history information can figure out a unique route in some sense. So, if we already have enough information to make a unique route in finite time/steps, we don't have to consider the former information. Another point of Markov localization is aprroximation. Namely, instead of achieve the highest accuracy in one step, it improves the accuracy gradually. 
```
bel(x_t) = p(x_t | z_{1:t}, u_{1:t}, m) 
         = p(x_t | z_t, z_{1:t-1}, u_{1:t}, m)
                // Bayes Rule
         = \frac{ p(z_t | x_t, z_{1:t-1}, u_{1:t}, m) * p(x_t | z_{1:t-1}, u_{1:t}, m) }{ p(z_t | z_{1:t-1}, u_{1:t}, m) }
                // denominator = p(z_t | z_{1:t-1}, u_{1:t}, m)
                // Markov Assumption
         = p(z_t | x_t) * p(x_t | z_{1:t-1}, u_{1:t}, m) / denominator
                // Total Probability
         = p(z_t | x_t) * \int [p(x_t|x_{t-1}, z_{1:t-1}, u_{1:t}, m)*p(x_{t-1}|z_{1:t-1}, u_{1:t}, m)] dx_{t-1} / denominator
                // Markov Assumption
         = p(z_t | x_t) * \int [p(x_t|x_{t-1}, u_t) * p(x_{t-1}|z_{1:t-1}, u_{1:t-1}, m)] dx_{t-1} / denominator
         = p(z_t | x_t) * \int [p(x_t|x_{t-1}, u_t) * bel(x_{t-1})] dx_{t-1} / denominator
         
posterior = measurement update (likelihood) * prediction
```

### Particle Filter

Particle Filter is one way to bring the above thought true. It approximate distribution of the position of our vehicle by the 'shape' of a group of 'particles', where each particle stands for a possible state of the vehicle. The higher density where particles are, the higher possibility our vehicle could be. 

Particle Filter still uses a dynamical way to improve its accuracy. Every time we obtain new measurements about the environment, the particle filter algorithm will use them and 'drive' particles to higher possibility places. 

How? **Resampling!!** At the beginning of prediction-measurement update cycle, every particle has a weight with value 1. Using new measurements, one can find a way to update those weights so that higer weights means higher possibility. Now we can do resampling on these particles according to the obtained new weights. In the prediction step, we use CTRV model.


## Control

### PID Controller

### Model Predictive Control





