# UDACITY: Self-Driving Car Nanodegree Program
## Project Submissions


### TERM 1
_Project 1: Lane Line Detection_ - Make use of Canny Edge Detection, image masking, and Hough Transforms to detect lane lines.

_Project 2: Traffic sign classifier_ - Deep Neural Networks w/ convolution, pooling, dropouts with image alterations (greyscale, normalization, center mean, etc.)

_Project 3: Behavioral Cloning_ - Autonomous driving car built from a convolutional neural network framework, trained on sample data taken from the Windows Driving Simulator.

_Project 4: Advanced Lane Line Detection_ - Use directional gradient and color channel thresholds and perspective transforms to predict lane lines with polynomial fits.

_Project 5: Vehicle Detection and Tracking_ - Trained a linear support vector machine with different feature extraction methods (spatial binning, color channel histograms, and histogram of orientation gradients (HOG)) to identify cars and track them through a video stream.


### TERM 2
_Project 6: Extended Kalman Filters_ - Implement an (Radar: Extended | Lidar: ) Kalman Filter to predict the state of a vehicle using sensor fusion of RADAR and LIDAR data feeds. Sensor covariance and noise estimates were provided.

_Project 7: Unscented Kalman FIlters_ - Implemented an (Radar: Unscented | Lidar: ) Kalman Filter to predict the state of a vehicle. The key difference to the EKF in project 6 is the UKF does not use a linear approximation for the process and radar measurement updates. Instead, the UKF utilizes point wise estimations for the mean and covariance, mapping through the state transition and computing the radar measurement residuals. The benefit to using this method is it allows for 1) non-linear process models, 2) second order derivate approximations to help account for noise, 3) maintains a Gaussian distribution. 

Tuning the linear and yaw angle accelearation rates was done manually through trail and error; however, the values can be approximately determined by understanding what object the model is supposed to measure. In this case, a bicycle can be expected to have a normally distributed linear acceleration noise with a mean of zero and standard deviation of 1 m/s^2, nu_a = N(0,1), and therefore 95% of all seen acceleration values will be +/-2 m/s^2. Similarly, the yaw acceleration noise is approximated by a normal distribution with a mean of zero and standard deviation of 0.75 rad/s^2, nu_yaw = N(0,0.75).

[Assumption: (linear acceleration noise) Average rider is moving at a constant speed of approx. 5 m/s and a standard deviation of 1m/s^2 would mean that 95% of all measurements falling withing 3-7 m/s^2 seems reasonable.]