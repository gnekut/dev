#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // LASER
  //measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ <<  1, 0, 0, 0,
              0, 1, 0, 0;
  //measurement covariance matrix
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // RADAR
  //measurement matrix (jacobian)
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;



}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    
    VectorXd x(4);
    MatrixXd Q(4,4);
    MatrixXd P(4,4);
    MatrixXd F(4,4);

    P <<  1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 100, 0,
          0, 0, 0, 100;

    F <<  1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      ekf_.H_ = tools.CalculateJacobian(x);
      ekf_.R_ = R_radar_;
     
      x <<  
      measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]),
      measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]),
      1,
      1;    

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.H_ = H_laser_;
      ekf_.R_ = R_laser_;

      x <<  
      measurement_pack.raw_measurements_[0], 
      measurement_pack.raw_measurements_[1], 
      1, 
      1;

    }

    ekf_.x_ = x;
    ekf_.P_ = P;
    ekf_.F_ = F;
    ekf_.Q_ = Q;


    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  const float n_ax = 9;
  const float n_ay = 9;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  ekf_.Q_ <<  
  dt_4*n_ax/4, 0, dt_3*n_ax/2, 0,
  0, dt_4*n_ay/4, 0, dt_3*n_ay/2,
  dt_3*n_ax/2, 0, dt_2*n_ax, 0,
  0, dt_3*n_ay/2, 0, dt_2*n_ay;

  ekf_.Predict(); 

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // Terminal Output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
