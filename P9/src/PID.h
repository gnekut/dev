#ifndef PID_H
#define PID_H

#include <vector>

class PID {
private:

  double p_cte; // Proportional cross-track error (pCTE - amplitude)
  double i_cte; // Integral cross-track error (iCTE - bias)
  double d_cte; // Differential cross-track error (dCTE - dampening)
  double dp_cte; // Previous CTE value (dpCTE)
  double sse_cte; // Sum-squared error of crosstrack error

public:
  
  int cnt; // Iterations 

  
  
  double p_k; // pCTE gain
  double i_k; // iCTE gain
  double d_k; // dCTE gain
  std::vector<double> k; // Gain vector

  double p_kd; // pCTE incremental gain multiplier
  double i_kd; // iCTE incremental gain multiplier
  double d_kd; // dCTE incremental gain multiplier
  std::vector<double> kd;


  // Constructor
  PID();

  // Deconstructor
  virtual ~PID();

   // Initialize PID.
  void Init(double Kp, double Ki, double Kd);

  // Update the PID error variables given cross track error.
  void Update(double cte);

  // Calculate the total PID correction.
  double Total();

};

#endif /* PID_H */
