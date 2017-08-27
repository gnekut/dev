#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &ests,
                              const vector<VectorXd> &gt) {
	
	// Initialize RMSE
	VectorXd rmse = VectorXd::Zero(4);
	if(ests.size() == gt.size() && ests.size() > 0){
		// Squared residuals summartion
		for(unsigned int i=0; i < ests.size(); ++i){
			VectorXd res = ests[i] - gt[i];
			res = res.array()*res.array();
			rmse += res;
		}
		rmse = rmse/ests.size(); // Average
		rmse = rmse.array().sqrt(); // Sqaure root

	} else {
		cout << "Invalid estimation or ground truth data" << endl;
	}
	return rmse;
}



