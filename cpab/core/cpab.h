#include <vector>

#ifndef CPAB_OPS_CPU
#define CPAB_OPS_CPU

int sign(int r);
bool cmpf(float x, float y, float eps = 1e-6f);
int get_cell(float x, const float xmin, const float xmax, const int nc);
float right_boundary(int c, const float xmin, const float xmax, const int nc);
float left_boundary(int c, const float xmin, const float xmax, const int nc);
float get_velocity(float x, const float* A, const float xmin, const float xmax, const int nc);
float get_psi(float x, float t, const float* A, const float xmin, const float xmax, const int nc);
float get_hit_time(float x, const float* A, const float xmin, const float xmax, const int nc);

float integrate_closed_form(float x, float t, const float* A, const float xmin, const float xmax, const int nc);
float get_numeric_phi(float x, float t, int nSteps2, const float* A, const float xmin, const float xmax, const int nc);
float integrate_numeric(float x, float t, const float* A, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2);

float integrate_closed_form_trace_full(float x, float t, const float* A, const float xmin, const float xmax, const int nc, std::vector<float> &xr, std::vector<float> &tr);
void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float xmin, const float xmax, const int nc);

float derivative_psi_theta(float x, float t, const int k, const int d, const float* B, const float* A, const float xmin, const float xmax, const int nc);
float derivative_phi_time(float x, float t, const float* A, const float xmin, const float xmax, const int nc);
float derivative_thit_theta(float x, const int k, const int d, const float* B, const float* A, const float xmin, const float xmax, const int nc);
float derivative_phi_theta_full(std::vector<float> &xr, std::vector<float> &tr, const int k, const int d, const float* B, const float* A, const float xmin, const float xmax, const int nc);

float derivative_phi_theta(float xini, float tm, int cm, const int k, const int d, const float* B, const float* A, const float xmin, const float xmax, const int nc);

#endif