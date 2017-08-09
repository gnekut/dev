#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &ests,
                              const vector<VectorXd> &gt) {
	// Root Mean Square Error (RMSE) initializer
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Estimates and true state must be non-zero, eqaul length
	if(ests.size() == gt.size() && ests.size() > 0){
		
		// Compute the sum of the squared residuals for [px, py, vx, vy]
		// Average and sqaure root
		for(unsigned int i=0; i < ests.size(); ++i){
			VectorXd res = ests[i] - gt[i];
			res = res.array()*res.array();
			rmse += res;
		}
		rmse = rmse/ests.size();
		rmse = rmse.array().sqrt();

	} else {
		cout << "Invalid estimation or ground truth data" << endl;
	}

	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	// Initialize
	MatrixXd Hj(3,4);
	float px = x_state[0];
	float py = x_state[1];
	float vx = x_state[2];
	float vy = x_state[3];

	// Pre-compute
	float px2_py2 = px*px+py*py;
	float px2_py2_sqrt = sqrt(px2_py2);

	// Jacobian (Hj) calculation
	// Div zero check [fabs(): absolute value, float]
	if(fabs(px2_py2) > 0.001){

		Hj << 
		(px/px2_py2_sqrt), (py/px2_py2_sqrt), 0, 0,
		-(py/px2_py2), (px/px2_py2), 0, 0,
		py*(vx*py - vy*px)/(px2_py2 * px2_py2_sqrt), px*(px*vy - py*vx)/(px2_py2 * px2_py2_sqrt), px/px2_py2_sqrt, py/px2_py2_sqrt;

	} else {
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;

		Hj << 
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;
	}

	return Hj;

}
