#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // Initialize MPC
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["throttle"];




          // ===================================== Transform and Compute Poly =======================================
          // Way-point transformation (global -> local coordinates)
          for (int i = 0; i < ptsx.size(); ++i) {
            double shift_x = ptsx[i] - px; 
            double shift_y = ptsy[i] - py;
            ptsx[i] = shift_x * cos(-psi) - shift_y * sin(-psi);
            ptsy[i] = shift_x * sin(-psi) + shift_y * cos(-psi);
          }

          // Compute polynomial coefficients (note: transform std::vector to Eigen)
          double* ptsx_ptr = &ptsx[0];
          Eigen::Map<Eigen::VectorXd> ptsx_eigen(ptsx_ptr, ptsx.size());
          double* ptsy_ptr = &ptsy[0];
          Eigen::Map<Eigen::VectorXd> ptsy_eigen(ptsy_ptr, ptsy.size());
          auto coeffs = polyfit(ptsx_eigen, ptsy_eigen, 3);

          // =======================================================================================




          
          // ================================ Current (latency) and Predicted States =============================
          Eigen::VectorXd state(6);
          double cte = coeffs[0];         // Current cross-track error (f(0))
          double epsi = -atan(coeffs[1]); // Current heading error (f'(0))

          const double Lf = 2.67;
          const double dt_late = 0.1; 
          const double y_late = 0.0;
          double x_late = v * dt_late;
          double psi_late = - v * delta * dt_late / Lf;
          double v_late = v + a * dt_late;
          double cte_late = cte + v * sin(epsi) * dt_late;
          double epsi_late = epsi + psi_late;                   
          state << x_late, y_late, psi_late, v_late, cte_late, epsi_late;

          vector<double> mpc_results;
          mpc_results = mpc.Solve(state, coeffs); // Predicted actuations and locations (delta, a, x<>, y<>)
          // =======================================================================================


          json msgJson;
          

          // ====================================== Output =======================================
          msgJson["steering_angle"] = mpc_results[0] / (Lf * deg2rad(25));  // Normalized steering angle actuation
          msgJson["throttle"] = mpc_results[1];                               // Acceleration actuation

          // Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          for (int i = 2; i < mpc_results.size(); i++) {
            if (i % 2 == 0) {
              mpc_x_vals.push_back(mpc_results[i]);
            } else {
              mpc_y_vals.push_back(mpc_results[i]);
            }
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;


          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for (int i = 0; i < 30; i++) {
            next_x_vals.push_back(i * 2);
            next_y_vals.push_back(polyeval(coeffs, i * 2));
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;
          // =======================================================================================







          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
