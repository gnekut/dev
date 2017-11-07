# UDACITY: Self-Driving Car Nanodegree Program
## Project Submissions


### TERM 1
_Project 1: Lane Line Detection_
Make use of Canny Edge Detection, image masking, and Hough Transforms to detect lane lines.

_Project 2: Traffic sign classifier_
Deep Neural Networks w/ convolution, pooling, dropouts with image alterations (greyscale, normalization, center mean, etc.)

_Project 3: Behavioral Cloning_
Autonomous driving car built from a convolutional neural network framework, trained on sample data taken from the Windows Driving Simulator.

_Project 4: Advanced Lane Line Detection_
Use directional gradient and color channel thresholds and perspective transforms to predict lane lines with polynomial fits.

_Project 5: Vehicle Detection and Tracking_
Trained a linear support vector machine with different feature extraction methods (spatial binning, color channel histograms, and histogram of orientation gradients (HOG)) to identify cars and track them through a video stream.


### TERM 2
_Project 6: Extended Kalman Filters_
Implement an (Radar: Extended | Lidar: ) Kalman Filter to predict the state of a vehicle using sensor fusion of RADAR and LIDAR data feeds. Sensor covariance and noise estimates were provided.


_Project 7: Unscented Kalman Filters_
Implemented an (Radar: Unscented | Lidar: ) Kalman Filter to predict the state of a vehicle. The key difference to the EKF in project 6 is the UKF does not use a linear approximation for the process and radar measurement updates. Instead, the UKF utilizes point wise estimations for the mean and covariance, mapping through the state transition and computing the radar measurement residuals. The benefit to using this method is it allows for 1) non-linear process models, 2) second order derivate approximations to help account for noise, 3) maintains a Gaussian distribution. 

Tuning the linear and yaw angle accelearation rates was done manually through trail and error; however, the values can be approximately determined by understanding what object the model is supposed to measure. In this case, a bicycle can be expected to have a normally distributed linear acceleration noise with a mean of zero and standard deviation of 3 m/s^2, nu_a = N(0,3), and therefore 95% of all seen acceleration values will be +/-6 m/s^2. Similarly, the yaw acceleration noise is approximated by a normal distribution with a mean of zero and standard deviation of 0.5 rad/s^2, nu_yaw = N(0,0.5).


_Project 8: Particle Filters_
Creation of a particle fitler algoritm to localize a vehicle on a map with observable, known landmarks. In theory, a distribution of particles is sampled around the initial vehicle location (determined by something like GPS) and are assigned a weight based upon how well each of them "fits" the observations made by the actual vehicles sensors to given landmarks. The weights are calcuated by associating each observation (mapped from the car's relative frame of reference to the particles global coordinates) to a landmark and calculating the Euclidean distance. A particle is given a greater weight the smaller the sum of distances is. Particles are then resampled based upon how close they match the observations/landmarks (i.e., weights), with the higher weights being sampled more.

_Project 9: PID Controllers_
Implement a PID (proportional/integral/differential) control algorithm for car simulator. The error value is provided by the simulator (cross-track error: distance from center) and outputs steering response value to correct cars position.