#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 0;
double dt = 0;

// State [x, y, psi, v, cte, e_psi]
// Accuations [delta, a]
int x_s = 0;
int y_s = 1*N;
int psi_s = 2*N;
int v_s = 3*N;
int cte_s = 4*N;
int epsi_s = 5*N;
int delta_s = 6*N; // len(1:N-1)
int a_s = 7*N-1; // len(1:N-1)


// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
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
    Inputs: [fg: vector containing cost contraints, note that fg[0] is the total cost, fg[1:N] are the actual constraint values], [vars: ]
    */


    // ========================= TOTAL COST ======================================
    fg[0] = 0;
    // Adding distance and heading error/cost.
    for (int t = 0; t < N; t++) {
      fg[0] += CppAD::pow(vars[cte_s + t], 2); // Cross-track error (distance from polynomial)
      fg[0] += CppAD::pow(vars[epsi_s + t], 2); // Heading error (difference in polynomial slope and car direction)
      // fg[0] += CppAD::pow(vars[v_s + t], 2);
    }
    // Adding steering and acceleration error/cost.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += CppAD::pow(vars[delta_s + t], 2); // Steering
      fg[0] += CppAD::pow(vars[a_s + t], 2); // Acceleration
    }
    // Adding change in steering and acceleration error/cost (prevent erradic state transitions)
    for (int t = 0; t < N - 2; t++) {
      fg[0] += CppAD::pow(vars[delta_s + t + 1] - vars[delta_s + t], 2);
      fg[0] += CppAD::pow(vars[a_s + t + 1] - vars[a_s + t], 2);
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
      AD<double> des_y_0 = coeffs[0] + coeffs[1] * x_0; // First-order approximation of ref. line (ptsx/ptsy are two element vectors)
      AD<double> des_psi_0 = CppAD::atan(coeffs[1]); // Steering angle

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
      fg[1 + epsi_s + t] = epsi_t - ((psi_0 - des_psi_0) + (v_0 * delta_0 * dt) / Lf);

    }
    // =======================================================================================





    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;


  size_t n_vars = (6 * N) + (2 * (N-1)); // Number of variable states
  size_t n_constraints = (6 * N) + 1; // Number of contraint states



  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }
  vars[x_s] = state[0];
  vars[y_s] = state[1];
  vars[psi_s] = state[2];
  vars[v_s] = state[3];


  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  for (int i = 0; i < N; i++) {
    // X
    vars_lowerbound[x_s + i] = -std::numeric_limits<double>::max();
    vars_upperbound[x_s + i] = std::numeric_limits<double>::max();
    // Y
    vars_lowerbound[y_s + i] = -std::numeric_limits<double>::max();
    vars_upperbound[y_s + i] = std::numeric_limits<double>::max();
    // PSI
    vars_lowerbound[psi_s + i] = 0;
    vars_upperbound[psi_s + i] = 3.14159; 
    // V
    vars_lowerbound[v_s + i] = 0;
    vars_upperbound[v_s + i] = 50;
    // CTE
    vars_lowerbound[cte_s + i] = -5;
    vars_upperbound[cte_s + i] = 5;
    // ePSI
    vars_lowerbound[epsi_s + i] = -0.78539; // -PI/4
    vars_upperbound[epsi_s + i] = 0.78539; // PI/4
    if (i < N - 1) {
      // Delta
      vars_lowerbound[delta_s + i] = -0.436332;
      vars_upperbound[delta_s + i] = 0.436332;
      // A
      vars_lowerbound[a_s + i] = -1;
      vars_upperbound[a_s + i] = 1;

    }
  }


  // TODO: Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

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

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  return {};
}
