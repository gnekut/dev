#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


/*
TO DO:
- Simplify all code to utilize array arithmetic where possible
*/


UKF::UKF() {
  // initialization boolean
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);
  P_(0,0) = 0.2;
  P_(1,1) = 0.2;
  P_(2,2) = 0.5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Laser covariance matrix, additive noise factors (m^2).
  R_laser_ = MatrixXd::Zero(2,2);
  R_laser_(0,0) = std_laspx_*std_laspx_;
  R_laser_(1,1) = std_laspy_*std_laspy_;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Radar covariance matrix, additive noise factors (m^2/s^2).
  R_radar_ = MatrixXd::Zero(3,3);
  R_radar_(0,0) = std_radr_*std_radr_;
  R_radar_(1,1) = std_radphi_*std_radphi_;
  R_radar_(2,2) = std_radrd_*std_radrd_;

  // Time when the state is true, in microseconds.
  time_us_ = 0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point count (based on project feedback)
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  X_sig_pred_ = MatrixXd(n_x_,n_sig_);

  // Sigma point weights
  weights_ = VectorXd::Constant(n_sig_, 0.5/(lambda_+n_aug_));
  weights_(0) = lambda_/(lambda_+n_aug_);
  
}


UKF::~UKF() {}

// =============================================================================
// =============================================================================
// =============================================================================

void UKF::ProcessMeasurement(MeasurementPackage m_pkg) {
  if (!is_initialized_) {
    if (m_pkg.sensor_type_ == MeasurementPackage::RADAR) {
      x_ << m_pkg.raw_measurements_[0] * cos(m_pkg.raw_measurements_[1]),
            m_pkg.raw_measurements_[0] * sin(m_pkg.raw_measurements_[1]),
            0,
            0,
            0;
    } else if (m_pkg.sensor_type_ == MeasurementPackage::LASER) {
      x_ << m_pkg.raw_measurements_[0], 
            m_pkg.raw_measurements_[1],
            0,
            0,
            0;
    }

    std::cout << R_radar_;
    is_initialized_ = true;
    time_us_ = m_pkg.timestamp_;

  } else {

    // =============================================================================
    // Update time difference
    double dt = (m_pkg.timestamp_ - time_us_) / 1000000.0;
    time_us_ = m_pkg.timestamp_;


    // Predict
    if (dt > 0.0001) {
      Prediction(dt);
    }
    

    // Update
    if ((m_pkg.sensor_type_ == MeasurementPackage::LASER) && use_laser_) {
      UpdateLidar(m_pkg);
    } else if((m_pkg.sensor_type_ == MeasurementPackage::RADAR) && use_radar_) {
      UpdateRadar(m_pkg);
    }
  }

}

// =============================================================================
// =============================================================================
// =============================================================================
void UKF::Prediction(double dt) {

  // Initialize augmented state vector
  VectorXd x_in = VectorXd::Zero(n_x_);
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug.tail(n_aug_ - n_x_) = VectorXd::Zero(n_aug_ - n_x_);
    
  // Initialize augmented covariance matrix
  MatrixXd P_in = MatrixXd::Zero(n_x_, n_x_);
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_aug_-2, n_aug_-2) = std_a_ * std_a_;
  P_aug(n_aug_-1, n_aug_-1) = std_yawdd_ * std_yawdd_;


  // Calculate sqrt(P_aug)
  MatrixXd A = P_aug.llt().matrixL();


  // =============================================================================
  // Initialize augmented sigma point matrix
  MatrixXd X_sig_aug = MatrixXd::Zero(n_aug_, n_sig_);
  X_sig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    X_sig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    X_sig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }

  // =============================================================================
  // Compute predicted state
  for (int i = 0; i < n_sig_; i++) {
    double px = X_sig_aug(0,i); // position-x
    double py = X_sig_aug(1,i); // position-y
    double v = X_sig_aug(2,i); // linear velocity
    double yaw = X_sig_aug(3,i); // yaw angle
    double yawd = X_sig_aug(4,i); // yaw angle change rate
    double nu_a = X_sig_aug(5,i); // linear acceleration noise
    double nu_yawdd = X_sig_aug(6,i); // angluar/yaw acceleration noise
    double px_p, py_p;

    /*
    Check for division by zero (if yawd = 0 has no change to it's yaw angle and therefore
    is moving in a straight line along velocity vector, v, and therefore is a simple
    geometric computation).
    */
    if (fabs(yawd) > 0.001) {
        px_p = px + v/yawd * (sin(yaw + yawd*dt) - sin(yaw));
        py_p = py + v/yawd * (cos(yaw) - cos(yaw+yawd*dt));
    }
    else {
        px_p = px + v*dt*cos(yaw);
        py_p = py + v*dt*sin(yaw);
    }

    // Transform and assign all sigma points through state transition
    X_sig_pred_(0,i) = px_p + 0.5*(dt*dt*cos(yaw)*nu_a);
    X_sig_pred_(1,i) = py_p + 0.5*(dt*dt*sin(yaw)*nu_a);
    X_sig_pred_(2,i) = v + dt*nu_a;
    X_sig_pred_(3,i) = yaw + dt*yawd + 0.5*(dt*dt*nu_yawdd);
    X_sig_pred_(4,i) = yawd + dt*nu_yawdd;
    
    x_in += weights_(i) * X_sig_pred_.col(i); // Compute predicted mean
  }
  
  x_in(3) = NormAngle(x_in(3)); // Normalize Angle
  
  // =============================================================================
  // Compute predicted covariance
  for (int i = 0; i < n_sig_; i++) {
    
    VectorXd x_diff = X_sig_pred_.col(i) - x_in; // State vector residuals
    x_diff(3) = NormAngle(x_diff(3)); // Normalize yaw angle
    P_in += weights_(i) * (x_diff) * (x_diff).transpose();
  }

  x_ = x_in; //
  P_ = P_in;

}



// =============================================================================
// =============================================================================
// =============================================================================
void UKF::UpdateLidar(MeasurementPackage m_pkg) {

  // Measurement vector initialization
  VectorXd z_meas(2);
  z_meas << m_pkg.raw_measurements_[0], 
            m_pkg.raw_measurements_[1];

  MatrixXd H(2, 5);
  H << 1, 0, 0, 0, 0, 
        0, 1, 0, 0, 0;

  VectorXd y = z_meas - H * x_; // y(2,1)
  MatrixXd P_Ht = P_ * H.transpose(); // P_Ht[5,2]
  MatrixXd S = H * P_Ht + R_laser_; // S[2,2]
  MatrixXd Si = S.inverse(); // Si[2,2]
  MatrixXd K = P_Ht * Si; // K[5,2]
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  x_ = x_ + K * y; // Update state
  P_ = (I - K * H) * P_; // Update covariance
  
  float nis;
  nis = y.transpose() * Si * y;
  std::cout << nis << std::endl;
}



// =============================================================================
// =============================================================================
// =============================================================================
void UKF::UpdateRadar(MeasurementPackage m_pkg) {

  // Measurement vector initialization
  VectorXd z_meas(3);
  z_meas << m_pkg.raw_measurements_[0], 
            m_pkg.raw_measurements_[1], 
            m_pkg.raw_measurements_[2];

  // Initialize radar covariance, noise, and state transformation structures
  MatrixXd S = MatrixXd::Zero(3,3);
  MatrixXd T = MatrixXd::Zero(n_x_,3);
  
  // Compute predicted mean (z_pred)
  VectorXd z_pred = VectorXd::Zero(3);
  MatrixXd Z_sig = MatrixXd::Zero(3,n_sig_);
  for (int i=0; i<n_sig_; i++) {

    double px = X_sig_pred_(0,i);
    double py = X_sig_pred_(1,i);
    double v = X_sig_pred_(2,i);
    double yaw = X_sig_pred_(3,i);

    // Transform state vector into radar measurement coordinates
    
    if (fabs(px) > 0.01 && fabs(py) > 0.01) {
      // Prevent division by zero
      Z_sig(0,i) = sqrt(px*px + py*py);
      Z_sig(1,i) = atan2(py, px);
      Z_sig(2,i)= (px*cos(yaw)*v + py*sin(yaw)*v) / Z_sig(0,i);
    }
    
  }

  // GN 2017-08-27: Update based on feedback (Zsig [3,15] * weights [15,1] --> therefore produces the correct matrix multiplication)
  z_pred = Z_sig * weights_; // Predicted mean state (transformed to radar measurement space)


  for (int i = 0; i < n_sig_; i++) {
    VectorXd z_sig_resid = Z_sig.col(i) - z_pred; // State sigma point residuals
    z_sig_resid(1) = NormAngle(z_sig_resid(1)); // Normalize state vector angle, theta
    S += weights_(i) * (z_sig_resid) * (z_sig_resid).transpose(); // Predicted covariance
    T += weights_(i) * (X_sig_pred_.col(i) - x_) * (z_sig_resid).transpose(); // Cross-correlation
  }


  S += R_radar_; // Add sensor measurement noise
  MatrixXd Si = S.inverse(); // Invert S
  MatrixXd K = T * Si; // Kalman gain

  VectorXd z_resid = z_meas - z_pred; // Measurement residuals
  z_resid(1) = NormAngle(z_resid(1)); // Normalize angle
  
  x_ += K * (z_resid); // Update state
  P_ -= K * S * K.transpose(); // Update covariance

}


double UKF::NormAngle(double x_in) {
  double x_out = fmod(x_in, 2*M_PI);
  if (x_out > M_PI) {
    x_out -= 2*M_PI;
  } else if (x_out < -M_PI) {
    x_out += 2*M_PI;
  }

  return x_out;
}