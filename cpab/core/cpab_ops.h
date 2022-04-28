#ifndef CPAB_OPS_CPU
#define CPAB_OPS_CPU

// FUNCTIONS
int get_cell(const float& x, const float& xmin, const float& xmax, const int& nc);
float get_velocity(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc);
float derivative_velocity_dx(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc);

// INTEGRATION
float integrate_closed_form(float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc);
void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc);
float integrate_numeric(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2);

// DERIVATIVE
void derivative_phi_theta(float* dphi_dtheta, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc);

// GRADIENT SPACE
float derivative_phi_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const float& xmin, const float& xmax, const int& nc);
void derivative_phi_x_theta(float* dphi_dx_dtheta, const float& xini, const float& tini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc);
float derivative_phi_x_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const float& xmin, const float& xmax, const int& nc);

#endif