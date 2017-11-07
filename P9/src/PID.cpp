#include "PID.h"
#include <math.h>

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	/*
	Function: Initialize all PID controller parameters
	Input(s): [Kp: proportional CTE gain], [Ki: integral CTE gain], [Kd: differential CTE gain]
	Output(s): N/A
	*/

	i_cte = 0.0; // Integral CTE
	dp_cte = 0.0; // Previous CTE (differential)
	sse_cte = 0.0; // Sum-squared error
	cnt = 0; // PID iteration counter

	// Cross-track error gains
	p_k = Kp;
	i_k = Ki;
	d_k = Kd;
	k.push_back(p_k);
	k.push_back(i_k);
	k.push_back(d_k);

	// Incremental gain multipliers
	p_kd = 0.5 * Kp;
	i_kd = 0.5 * Ki;
	d_kd = 0.5 * Kd;
	kd.push_back(p_kd);
	kd.push_back(i_kd);
	kd.push_back(d_kd);

}

void PID::Update(double cte) {
	/*
	Function: Update cross-track error parameters for proportional, integral, and differential states.
	Input(s): [cte: Current cross-track error measurement]
	Output(s): N/A [internal PID values]
	*/

	p_cte = cte; 
	i_cte += cte;
	d_cte = cte - dp_cte;
	dp_cte = cte;
	sse_cte = pow(cte, 2.0);
	cnt += 1;
	
}


double PID::Total() {
	/*
	Function: Compute total cross-track error effect
	Input(s): N/A [internal PID values]
	Output(s): [pid_out: Total error effect from PID components]
	*/

	double pid_out;
	pid_out = -(p_k * p_cte + i_k * i_cte + d_k * d_cte);
	return pid_out;
}