#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;



int N = 10;       // Number of variable states ([x, y, psi, v, cte, e_psi] * N)
double dt = 0.1; // Time duration between states


// Vector Indices (start of state vectors 'i')
int x_s = 0;
int y_s = x_s + N;
int psi_s = y_s + N;
int v_s = psi_s + N;
int cte_s = v_s + N;
int epsi_s = cte_s + N;
int delta_s = epsi_s + N; // len(1:N-1)
int a_s = delta_s + N-1;  // len(1:N-1)


/* 
This value assumes the model presented in the classroom is used. It was obtained by measuring 
the radius formed by running the vehicle in the simulator around in a circle with a constant 
steering angle and velocity on a flat terrain. Lf was tuned until the the radius formed by the 
simulating the model presented in the classroom matched the previous radius. This is the length 
from front to CoG that has a similar radius.
*/ 
const double Lf = 2.67;



class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    /*
    Function: Compute the total cost and constraints
    Inputs: 
      [fg: vector containing cost contraints, note that fg[0] is the total cost, fg[1:N] are the actual constraint values]
      [vars: state variable values to adjust within contrainsts to minimize cost function]
    */


    // ========================= TOTAL COST ======================================
    // State reference values: Adjusts the cost function(s) to center around the reference value
    double cte_ref  = 0;
    double epsi_ref = 0;
    double v_ref = 70;

    fg[0] = 0;
    // Adding distance and heading error/cost.
    for (int t = 0; t < N; t++) {
      fg[0] += 1000 * CppAD::pow(vars[cte_s + t] - cte_ref, 2);   // Cross-track error (distance from polynomial)
      fg[0] += 10000 * CppAD::pow(vars[epsi_s + t] - epsi_ref, 2); // Heading error (difference in polynomial slope and car direction)
      fg[0] += 100 * CppAD::pow(vars[v_s + t] - v_ref, 2);       // Velocity error
    }
    // Adding steering and acceleration error/cost.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += 100 * CppAD::pow(vars[delta_s + t], 2); // Steering angle penalty (straight vs. turning)
      fg[0] += 1 * CppAD::pow(vars[a_s + t], 2);     // Acceleration penalty (constant speed vs. changing)
    }
    // Adding change in steering and acceleration error/cost (prevent erradic state transitions)
    for (int t = 0; t < N - 2; t++) {
      fg[0] += 10000 * CppAD::pow(vars[delta_s + t + 1] - vars[delta_s + t], 2); // Steering angle change (sharp vs smooth turns)
      fg[0] += 100 * CppAD::pow(vars[a_s + t + 1] - vars[a_s + t], 2);          // Acceleration change (hard vs soft accelerations)
    }
    // =============================================================================





    // ====================== CONSTRAINTS ===========================================
    // Initialization
    fg[1 + x_s] = vars[x_s];
    fg[1 + y_s] = vars[y_s];
    fg[1 + psi_s] = vars[psi_s];
    fg[1 + v_s] = vars[v_s];
    fg[1 + cte_s] = vars[cte_s];
    fg[1 + epsi_s] = vars[epsi_s];

    for (int t=1; t<N; t++) {
      // Current states (t)
      AD<double> x_0 = vars[x_s + t - 1];
      AD<double> y_0 = vars[y_s + t - 1];
      AD<double> psi_0 = vars[psi_s + t - 1];
      AD<double> v_0 = vars[v_s + t - 1];
      AD<double> cte_0 = vars[cte_s + t - 1];
      AD<double> epsi_0 = vars[epsi_s + t - 1];
      AD<double> delta_0 = vars[delta_s + t - 1];
      AD<double> a_0 = vars[a_s + t - 1];
      
      // Computed desired state
      AD<double> des_y_0 = coeffs[0] + (coeffs[1] * x_0) + (coeffs[2] * x_0 * x_0) + (coeffs[3] * x_0 * x_0 * x_0); // Second-order approximation of ref. line (ptsx/ptsy are two element vectors)
      AD<double> des_psi_0 = CppAD::atan(coeffs[1] + (2 * coeffs[2] * x_0) + (3 * coeffs[3] * x_0 * x_0));          // Steering angle

      // Future states (t+1)
      AD<double> x_t = vars[x_s + t];
      AD<double> y_t = vars[y_s + t];
      AD<double> psi_t = vars[psi_s + t];
      AD<double> v_t = vars[v_s + t];
      AD<double> cte_t = vars[cte_s + t];
      AD<double> epsi_t = vars[epsi_s + t];

      fg[1 + x_s + t] = x_t - (x_0 + (v_0 * CppAD::cos(psi_0) * dt));
      fg[1 + y_s + t] = y_t - (y_0 + (v_0 * CppAD::sin(psi_0) * dt));
      fg[1 + psi_s + t] = psi_t - (psi_0 - (v_0 * delta_0 * dt) / Lf); // NOTE: made (psi_0 + v_0 ...) to (psi_o - v_0 ...) as suggested
      fg[1 + v_s + t] = v_t - (v_0 + (a_0 * dt));
      fg[1 + cte_s + t] = cte_t - ((des_y_0 - y_0) + (v_0 * CppAD::sin(epsi_0) * dt));
      fg[1 + epsi_s + t] = epsi_t - ((psi_0 - des_psi_0) - (v_0 * delta_0 * dt) / Lf);

    }
    // =======================================================================================


  }
};

// MPC Contructor
MPC::MPC() {}

// MPC Deconstructor
MPC::~MPC() {}


vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;




  // ================================= Initialization ========================================
  int n_constraints = (6 * N);              // Number of contraint variable states (6 variables, N variable states)
  int n_vars = n_constraints + (2 * (N-1)); // Number of variable values (2N-1 is for the actuator variable values)
  
  double x_i = state[0];
  double y_i = state[1];
  double psi_i = state[2];
  double v_i = state[3];
  double cte_i = state[4];
  double epsi_i = state[5];

  // Initialize all state variables
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  // Initialize upper/lower constainst bounds
  // [Note that each starting constraint is given the value of the current state.]
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0.0;
    constraints_upperbound[i] = 0.0;
  }
  constraints_lowerbound[x_s] = x_i;
  constraints_upperbound[x_s] = x_i;
  constraints_lowerbound[y_s] = y_i;
  constraints_upperbound[y_s] = y_i;
  constraints_lowerbound[psi_s] = psi_i;
  constraints_upperbound[psi_s] = psi_i;
  constraints_lowerbound[v_s] = v_i;
  constraints_upperbound[v_s] = v_i;
  constraints_lowerbound[cte_s] = cte_i;
  constraints_upperbound[cte_s] = cte_i;
  constraints_lowerbound[epsi_s] = epsi_i;
  constraints_upperbound[epsi_s] = epsi_i;
  
  
  // Initialize upper/lower variable value bounds
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  for (int i = 0; i < delta_s; i++) {
    vars_lowerbound[i] = -std::numeric_limits<double>::max();
    vars_upperbound[i] =  std::numeric_limits<double>::max();
  }
  // Steering angle bounds (+/- 25 degrees)
  for (int i = delta_s; i < a_s; i++){
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] =  0.436332;
  }
  // Acceleration bounds (+/- 1.0)
  for (int i = a_s; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] =  1.0;
  }
  // =======================================================================================






  // ================================== Solver =============================================
  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  std::cout << "Cost: " << solution.obj_value << std::endl;
  // =======================================================================================




  // ====================================== Return Values ==================================
  std::vector<double> mpc_return;
  mpc_return.push_back(solution.x[delta_s]);
  mpc_return.push_back(solution.x[a_s]);
  for (int i = 0; i < N; i++) {
    mpc_return.push_back(solution.x[x_s + i]);
    mpc_return.push_back(solution.x[y_s + i]);
  }
  return mpc_return;
  // =======================================================================================

}
