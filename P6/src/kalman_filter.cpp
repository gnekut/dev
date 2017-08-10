#include "kalman_filter.h"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  /*
    x' = Fx + u (Note, u is Gaussian ~ N(0,sigma) so average effect on state is zero and the noise is accounted for in the process covariance, Q)
    P' = FP(F_t) + Q
  */

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  /* 
    Update: LIDAR
    z = [p_x, p_y, v_x, v_y] (measured values)
    y = z - H x
    S = H P H_t + R
    K = P H_t S^-1
    x' = x + K y
    P' = (I - K H) * P
  */

  // Process Measurement
  VectorXd y = z - H_ * x_;

  PushUpdate(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
    Update: RADAR
    y = z - h(x') [h(x): function to translate cartesian to polar coordinates]
    S = Hj P Hj_t + R
    K = P Hj_t S^-1
    x' = x + K y
    P' = (I - K Hj) * P
  */


  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  float px2_py2 = px*px + py*py;

  // Covert x_cart to x_polar
  VectorXd x_polar(3);
  if (fabs(px2_py2) > 0.00001) {
    x_polar << sqrt(px2_py2),
              atan2(py, px),
              (px*vx + py*vy)/sqrt(px2_py2);
  } else {
    x_polar << 0, 0, 0;
  }

  // Normalize and Process Measurement
  VectorXd y = z - x_polar; // [3]
  y(1) = fmod(y(1), M_PI);
  if (y(1) > M_PI) {
    y(1) = y(1) - 2*M_PI;
  } else if (y(1) < -1*M_PI) {
    y(1) = y(1) + 2*M_PI;
  }

  PushUpdate(y);

}

void KalmanFilter::PushUpdate(VectorXd &y) {
  // Compute Kalman Gain
  MatrixXd P_Ht = P_ * H_.transpose();
  MatrixXd S = H_ * P_Ht + R_; // [3,3]
  MatrixXd K = P_Ht * S.inverse(); // [4,3]
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size()); // [4,4]

  // Update State
  x_ = x_ + K * y; // [4] + [4,3] * [3]
  P_ = (I - K * H_) * P_; // ([4,4] - [4,3]*[3,4]) * [4,4]
}