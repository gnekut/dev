/*
GN 2017-10-23: Thoughts of possible errors:
1) Yaw angle, theta being (+/-) w.t.r. x-axis coordinates.
2) 

 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles_ = 10; // Initialize particle count

	// Noise Generators
	default_random_engine gen; // Generate uniform dist. numbers
	normal_distribution<double> dist_x(x, std[0]); // X-Gaussian Object
	normal_distribution<double> dist_y(y, std[1]); // Y-Gaussian Object
	normal_distribution<double> dist_theta(theta, std[2]); // Theta-Gaussian Object

	// Initialize Particles
	for (int i=0; i<num_particles_; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0/num_particles_;

		particles_.push_back(particle);
		weights_.push_back(particle.weight);
	}

	is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen; // Generate uniform dist. numbers

	// Predict Particles Position
	for (int p=0; p<num_particles_; p++){

		double x_temp;
		double y_temp;
		double theta_temp;

		// Zero-yaw rate separation
		if (fabs(yaw_rate) > 0.0001){
			x_temp = particles_[p].x + (velocity/yaw_rate) * (sin(particles_[p].theta + yaw_rate * delta_t) - sin(particles_[p].theta));
			y_temp = particles_[p].y + (velocity/yaw_rate) * (cos(particles_[p].theta) - cos(particles_[p].theta + yaw_rate * delta_t));
			theta_temp = particles_[p].theta + (yaw_rate * delta_t);
		} else {
			x_temp =  particles_[p].x + (velocity * cos(particles_[p].theta)* delta_t);
			y_temp =  particles_[p].y + (velocity * sin(particles_[p].theta)* delta_t);
			theta_temp = particles_[p].theta;
		}

		// Generate noise distributions
		normal_distribution<double> dist_x(x_temp, std_pos[0]); // X-Gaussian Object
		normal_distribution<double> dist_y(y_temp, std_pos[1]); // Y-Gaussian Object
		normal_distribution<double> dist_theta(theta_temp, std_pos[2]); // Theta-Gaussian Object

		// Generate new particle informations w/ noise
		particles_[p].x = dist_x(gen);
		particles_[p].y = dist_y(gen);
		particles_[p].theta = dist_theta(gen);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {

	int ob_size = observations.size();
	
	// Pre-compute variables
	double weights_normalizer = 0.0; // Sum of all weights for normalization step
	double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]); // Gaussian normalizer
	double x_denom = (2.0 * std_landmark[0] * std_landmark[0]); // Gaussian X-demoninator
	double y_denom = (2.0 * std_landmark[1] * std_landmark[1]); // Gaussian Y-demoninator


	// ========== PARTICLE ITERATION ==========
	for (int p=0; p<num_particles_; p++) {

		// Reset particles associations and weight
		particles_[p].associations.clear();
		particles_[p].sense_x.clear();
		particles_[p].sense_y.clear();
		particles_[p].weight = 1.0;


		// =========== LANDMARK EXTRACTION ========== 
		// Remap Map struct to LandmarkObs struct and ensure landmark is within sensor range for particle
		std::vector<LandmarkObs> landmarks;
		for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
			LandmarkObs lm;
			lm.id = map_landmarks.landmark_list[i].id_i;
			lm.x = map_landmarks.landmark_list[i].x_f;
			lm.y = map_landmarks.landmark_list[i].y_f;
			if (dist(lm.x, lm.y, particles_[p].x, particles_[p].y) < sensor_range) {
				landmarks.push_back(lm);
			}
		}


		// ============= COORDINATE TRANSFORMATION ==========
		// Transform local car observations to global reference frame (partible position + observation)
		std::vector<LandmarkObs> observations_p;
		for (int i=0; i<ob_size; i++) {
			LandmarkObs obs;
			obs.x = particles_[p].x + (observations[i].x * cos(particles_[p].theta)) - (observations[i].y * sin(particles_[p].theta));
			obs.y = particles_[p].y + (observations[i].x * sin(particles_[p].theta)) + (observations[i].y * cos(particles_[p].theta));
			observations_p.push_back(obs);
		}


		


		// ========== LANDMARK-OBSERVATION ASSOCIATION ==========
		// Return (observations_p) for only those observations with an associated landmark
		for (int j=0; j<ob_size; j++) {
			double least_dist = dist(landmarks[0].x, landmarks[0].y, observations_p[j].x, observations_p[j].y); // Init. distance
			observations_p[j].id = landmarks[0].id; // Init. closest ID
			for (int i=0; i<landmarks.size(); i++) {
				double obs_dist = dist(landmarks[i].x, landmarks[i].y, observations_p[j].x, observations_p[j].y);
				if (least_dist > obs_dist) {
					observations_p[j].id = landmarks[i].id;
					least_dist = obs_dist;
				}
			}
			particles_[p].associations.push_back(observations_p[j].id);
			particles_[p].sense_x.push_back(observations_p[j].x);
			particles_[p].sense_y.push_back(observations_p[j].y);
			cout << "[P:" << p << ", O:" << j << ", A:" << observations_p[j].id <<"]\n";
		}



		/*
		// Print associations and coordinates
		cout << "P[" << p << "], Associations: ";
		for (int i=0; i<particles_[p].associations.size();i++) {
			cout << "[A: " << particles_[p].associations[i] << " ";
			cout << ", X: " << particles_[p].sense_x[i] << " ";
			cout << ", Y: " << particles_[p].sense_y[i] << "]";
		}
		cout << "\n";
		*/


		// ========== WEIGHT CALCULATIONS ============
		
		for (int i=0; i<particles_[p].associations.size(); i++) {
			int obs_lm_id = particles_[p].associations[i];
			double obs_x = particles_[p].sense_x[i];
			double obs_y = particles_[p].sense_y[i];
			
			// Iterate over all landmarks to find associated.
			for (int l=0; l<landmarks.size(); l++) {
				LandmarkObs lm = landmarks[l];
				if (lm.id == obs_lm_id) {
					double x_num = pow((obs_x-lm.x), 2.0);
					double y_num = pow((obs_y-lm.y), 2.0);
					particles_[p].weight *= (gauss_norm) * exp(-1.0*((x_num/x_denom)+(y_num/y_denom)));
					//cout << "  Weight[" << p << ", " << l << "]:  " << particles_[p].weight << "\n";
				}
			}

		}
		weights_normalizer += particles_[p].weight; // Particle weight normalizer
	}


	// ========== NORMALIZE WEIGHTS ==========
	weights_.clear();

	for (int p=0; p<num_particles_; p++) {
		particles_[p].weight /= weights_normalizer;
		weights_.push_back(particles_[p].weight);
		//cout << "  Norm. Weight[" << p << "]:  " << particles_[p].weight << "\n";
	}
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    default_random_engine gen;
    std::discrete_distribution<int> weight_dist(weights_.begin(), weights_.end());


    std::vector<Particle> particles_new;
    particles_new.resize(num_particles_);
    for (int p=0; p<num_particles_; p++) {
        int p_i = weight_dist(gen);
        particles_new[p_i] = particles_[p_i];
    }
    particles_ = particles_new;

}


string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
