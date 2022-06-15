#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>

// FUNCTIONS

#define eps FLT_EPSILON;
#define inf INFINITY;

__device__ int sign(const int r){
    return (r > 0) - (r < 0);
}

__device__ int signf(const float r){
    return (r > 0) - (r < 0);
}

__device__ bool cmpf(float x, float y){
    return fabs(x - y) < eps;
}

__device__ bool cmpf0(const float& x){
    return fabs(x) < eps;
}

__device__ float right_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

__device__ float left_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    return xmin + c * (xmax - xmin) / nc - eps;
}

__device__ int get_cell(const float& x, const float& xmin, const float& xmax, const int& nc){
    int c = floor((x - xmin) / (xmax - xmin) * nc);
    c = max(0, min(c, nc-1));
    return c;
}

__device__ float get_velocity(const float& x, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    return a*x + b;
}

__device__ float get_velocity_dx(const float& x, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[(2*c) * n_batch + batch_index];
    return a;
}

// INTEGRATION CLOSED FORM

__device__ float get_psi(const float& x, const float& t,  const float& a, const float& b){
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return exp(t*a) * (x + (b/a)) - (b/a);
    }
}

__device__ float get_hit_time(float x, int c, const float& a, const float& b, const float& xmin, const float& xmax, const int& nc, float& xc, int& cc){

    const float v = a * x + b;
    if(cmpf0(v)) return inf;

    cc = c + signf(v);
    if(cc < 0 || cc >= nc) return inf;
    xc = (v > 0) ? right_boundary(c, xmin, xmax, nc) : left_boundary(c, xmin, xmax, nc);

    const float vc = a * xc + b;
    if(cmpf0(vc)) return inf;
    if(signf(v) != signf(vc)) return inf;
    if(xc == xmin || xc == xmax) return inf;

    if(cmpf0(a)){
        return (xc - x)/b;
    }else{
        return std::log(vc / v) / a;
    }
}

__device__ float integrate_closed_form(float x, float t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[(2*c) * n_batch + batch_index];
        b = A[(2*c+1) * n_batch + batch_index];

        thit = get_hit_time(x, c, a, b, xmin, xmax, nc, xc, cc);
        if (thit > t){
            return get_psi(x, t, a, b);
        }

        x = xc;
        c = cc;
        t -= thit;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return -1;
}

__device__ void integrate_closed_form_trace(float* result, float x, float t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[(2*c) * n_batch + batch_index];
        b = A[(2*c+1) * n_batch + batch_index];

        thit = get_hit_time(x, c, a, b, xmin, xmax, nc, xc, cc);
        if (thit > t){
            result[0] = get_psi(x, t, a, b);
            result[1] = t;
            result[2] = c;
            return;
        }

        x = xc;
        c = cc;
        t -= thit;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
}

// INTEGRATION NUMERIC

__device__ float get_psi_numeric(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    // const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return std::exp(t*a) * (x + (b/a)) - (b/a);
    }
}

__device__ float get_phi_numeric(const float& x, const float& t, const int& nSteps2, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    for(int j = 0; j < nSteps2; j++) {
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, n_batch, batch_index, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return yn;
}

__device__ float integrate_numeric(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_numeric(xPrev, c, deltaT, A, n_batch, batch_index);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric(xPrev, deltaT, nSteps2, A, n_batch, batch_index, xmin, xmax, nc);
            c = get_cell(xPrev, xmin, xmax, nc);
        }
    }
    return xPrev;
}


// DERIVATIVE

__device__ void derivative_psi_theta(double* gradpoints, const float& x, const int& c, const float& t, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    
    if (cmpf0(a)){
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            gradpoints[batch_index*(n_points * d) + point_index*d + k] += t*(x*ak + bk);
        }
    }
    else{
        const double tmp = exp(t*a);
        const double tmp1 = t * tmp * (x + b/a);
        const double tmp2 = (tmp-1)/pow(a, 2.0);
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            gradpoints[batch_index*(n_points * d) + point_index*d + k] += ak * tmp1 + tmp2 * (bk*a - ak*b);
        }
    }
}

__device__ float derivative_phi_time(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    if (cmpf0(a)){
        return b;
    }
    else{
        return exp(t*a)*(a*x + b);
    }
}

__device__ void derivative_thit_theta(double* gradpoints, const float& x, const int& c, const float& xc, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    if (cmpf0(a)){
        const double tmp = (x-xc) / pow(b, 2.0);
        for(int k=0; k < d; k++){
            const double bk = B[(2*c+1)*d + k];
            gradpoints[batch_index*(n_points * d) + point_index*d + k] -= tmp*bk;
        }
    }
    else{
        const double tmp1 = log( (a*xc + b) / (a*x + b) )/pow(a, 2.0);
        const double tmp2 = (x - xc) / (a * (a*x + b) * (a*xc + b) );
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];

            const double d1 = - ak * tmp1;
            const double d2 = ( bk*a - ak*b) * tmp2;;
            gradpoints[batch_index*(n_points * d) + point_index*d + k] -= d1 + d2;
        }
    }
}

__device__ void derivative_phi_theta(double* gradpoints, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            derivative_thit_theta(gradpoints, xm, c, xc, d, B, A, n_batch, batch_index, n_points, point_index);
            xm = xc;
        }
    }

    const float dpsi_dtime = derivative_phi_time(xm, cm, tm, A, n_batch, batch_index);
    for(int k=0; k < d; k++){
        gradpoints[batch_index*(n_points * d) + point_index*d + k] *= dpsi_dtime;
    }
    derivative_psi_theta(gradpoints, xm, cm, tm, d, B, A, n_batch, batch_index, n_points, point_index);
    
}


// KERNELS

__global__ void kernel_get_cell(
    const int n_points, const float* x, 
    const float xmin, const float xmax, const int nc, int* newpoints){  

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(point_index < n_points) {
        newpoints[point_index] = get_cell(x[point_index], xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_get_velocity(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = get_velocity(x[batch_index * n_points + point_index], A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_derivative_velocity_dx(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = get_velocity_dx(x[batch_index * n_points + point_index], A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_integrate_numeric(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float t, const float xmin, const float xmax, const int nc, 
    const int nSteps1, const int nSteps2, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_numeric(x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc, nSteps1, nSteps2);
    }
    return;
}

__global__ void kernel_integrate_closed_form(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float t, const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_closed_form(x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_derivative_closed_form(
    const int n_points, const int n_batch, const int d,
    const float* x, const float* A, const float* B, 
    const float t, const int xmin, const int xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch){ 
        float result[e];
        integrate_closed_form_trace(result, x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
            
        // float phi = result[0];
        float tm = result[1];
        int cm = result[2];
        derivative_phi_theta(gradpoints, x[batch_index * n_points + point_index], tm, cm, d, B, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
        
    }
    return;
}

__global__ void kernel_integrate_closed_form_trace(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float t, const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch) {
        float result[e];
        integrate_closed_form_trace(result, x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
        for(int p = 0; p < e; p++){
            newpoints[batch_index*(n_points * e) + point_index*e + p] = result[p];
        }
    }
    return;
}

__global__ void kernel_derivative_closed_form_trace(
    const int n_points, const int n_batch, const int d,
    const float* newpoints, const float* x, const float* A, const float* B, 
    const float xmin, const float xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    const int e = 3;

    if(point_index < n_points && batch_index < n_batch) {
        // float phi = newpoints[batch_index*(n_points * e) + point_index*e + 0];
        float tm = newpoints[batch_index*(n_points * e) + point_index*e + 1];
        int cm = newpoints[batch_index*(n_points * e) + point_index*e + 2];
        
        derivative_phi_theta(gradpoints, x[batch_index * n_points + point_index], tm, cm, d, B, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
    }
    return;
}



// INTERPOLATE
__device__ float clip(int num, int lower, int upper) {
    return max(lower, min(num, upper));
}
  
__device__ float interpolate_grid_forward(const float* x, const int& n_points, const int& batch_index, const int& point_index){
    
    float xc = x[batch_index*n_points + point_index]*(n_points - 1);
    int x0 = (int) std::floor(xc);
    int x1 = x0 + 1;
    x0 = clip(x0, 0, n_points-1);
    x1 = clip(x1, 0, n_points-1);
    float y0 = x[batch_index*n_points + x0];
    float y1 = x[batch_index*n_points + x1];
    float xd = (float) xc - x0;

    return y0 * (1 - xd) + y1 * xd;
}


__global__ void kernel_interpolate_grid_forward(
    const int n_points, const int n_batch, const float* x, float* y){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        y[batch_index * n_points + point_index] = interpolate_grid_forward(x, n_points, batch_index, point_index);
    }
    return;
}

__device__ void interpolate_grid_backward(float* gradient, const float* g, const float* x, const int& n_points, const int& batch_index, const int& point_index){
    
    int pos = n_points*batch_index + point_index;
            
    float xc = x[pos]*(n_points - 1);
    int x0 = (int) std::floor(xc);
    int x1 = x0 + 1;
    x0 = clip(x0, 0, n_points-1);
    x1 = clip(x1, 0, n_points-1);
    float y0 = x[batch_index*n_points + x0];
    float y1 = x[batch_index*n_points + x1];
    float xd = (float) xc - x0;

    gradient[n_points*batch_index + x0] += (1-xd) * g[pos];
    gradient[n_points*batch_index + x1] += xd * g[pos];
    gradient[n_points*batch_index + point_index] += (n_points-1)*(y1-y0) * g[pos];
}


__global__ void kernel_interpolate_grid_backward(
    const int n_points, const int n_batch, const float* g, const float* x, float* gradient){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        interpolate_grid_backward(gradient, g, x, n_points, batch_index, point_index);
    }
    return;
}


// GRADIENT SPACE

__device__ float derivative_thit_x(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    return  1.0 / (a*x + b);
}

__device__ float derivative_psi_x(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    // const float b = A[(2*c+1) * n_batch + batch_index];
    return  exp(t*a);
}

__device__ float derivative_psi_t(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    return  exp(t*a)*(a*x + b);
}

__device__ float derivative_phi_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dpsi_dx = 0.0;
    float dthit_dx = 0.0;
    if (cini == cm){
        dpsi_dx = derivative_psi_x(xini, cini, tini, A, n_batch, batch_index);
    }else{
        dthit_dx = derivative_thit_x(xini, cini, tini, A, n_batch, batch_index);
    }

    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            xm = xc;
        }
    }

    float dpsi_dtime = derivative_psi_t(xm, cm, tm, A, n_batch, batch_index);
    float dphi_dx = dpsi_dx + dpsi_dtime * dthit_dx;
   
    return dphi_dx;
    
}

__global__ void kernel_derivative_space_closed_form(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float t, const int xmin, const int xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch){ 
        float result[e];
        integrate_closed_form_trace(result, x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
            
        // float phi = result[0];
        float tm = result[1];
        int cm = result[2];
        float dphi_dx = derivative_phi_x(x[batch_index * n_points + point_index], t, tm, cm, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
        
        gradpoints[batch_index * n_points + point_index] = dphi_dx;
    }
    return;
}

// GRADIENT SPACE DERIVATIVE THETA

__device__ float derivative_psi_x_theta(const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    // const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    // const double bk = B[(2*c+1)*d + k];

    return t * exp(t*a) * ak;
}

__device__ float derivative_thit_x_theta(const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    return - (x*ak + bk)/std::pow(a*x + b, 2.0);
}

__device__ float derivative_psi_t_theta(const float& dtm, const float& x, const int& c, const float& t, const float* A, const int& k, const int& d, const float* B, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    return exp(t*a) * ( a*(a*x+b)*dtm + ak*(t*(a*x+b) + x) + bk);
}

__device__ float derivative_thit_theta_alt(const float& x, const int& c, const float& xc, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    if (cmpf0(a)){
        const double tmp = (x-xc) / pow(b, 2.0);
        return -(tmp*bk);
    }
    else{
        const double tmp1 = log( (a*xc + b) / (a*x + b) )/pow(a, 2.0);
        const double tmp2 = (x - xc) / (a * (a*x + b) * (a*xc + b) );

        const double d1 = - ak * tmp1;
        const double d2 = ( bk*a - ak*b) * tmp2;
        return -(d1+d2);
    }
}

__device__ void derivative_phi_x_theta(double* gradpoints, const float& xini, const float& tini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    
    for(int k=0; k < d; k++){
        float xm = xini;
        float dthit_dtheta_cum = 0.0;
        if (cini != cm){
            float xc;
            const int step = sign(cm - cini);
            for (int c = cini; step*c < cm*step; c += step){
                if (step == 1){
                    xc = right_boundary(c, xmin, xmax, nc);
                }else if (step == -1){
                    xc = left_boundary(c, xmin, xmax, nc);
                }
                dthit_dtheta_cum += derivative_thit_theta_alt(xm, c, xc, k, d, B, A, n_batch, batch_index);
                xm = xc;
            } 
        }
    
        float dpsi_dtime = derivative_psi_t(xm, cm, tm, A, n_batch, batch_index);

        float dthit_dx = 0.0;
        float dpsi_dx_dtheta = 0.0;
        float dthit_dx_dtheta = 0.0;
        if (cini == cm){
            dpsi_dx_dtheta = derivative_psi_x_theta(xini, cini, tini, A, k, d, B, n_batch, batch_index);
        }else{
            dthit_dx = derivative_thit_x(xini, cini, tini, A, n_batch, batch_index);
            dthit_dx_dtheta = derivative_thit_x_theta(xini, cini, tini, A, k, d, B, n_batch, batch_index);
        }
        float dpsi_dtime_dtheta = derivative_psi_t_theta(dthit_dtheta_cum, xm, cm, tm, A, k, d, B, n_batch, batch_index);
        float dphi_dx_dtheta = dpsi_dx_dtheta + dpsi_dtime_dtheta * dthit_dx + dpsi_dtime * dthit_dx_dtheta;
        gradpoints[batch_index*(n_points * d) + point_index*d + k] = dphi_dx_dtheta;
    }
}



__global__ void kernel_derivative_space_closed_form_dtheta(
    const int n_points, const int n_batch, const int d,
    const float* x, const float* A, const float* B, const float t,
    const float xmin, const float xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;

    const int e = 3;

    if(point_index < n_points && batch_index < n_batch) {
        float result[e];
        integrate_closed_form_trace(result, x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
            
        // float phi = result[0];
        float tm = result[1];
        int cm = result[2];
        
        derivative_phi_x_theta(gradpoints, x[batch_index * n_points + point_index], t, tm, cm, d, B, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
    }
    return;
}




// GRADIENT SPACE DERIVATIVE X

__device__ float derivative_thit_x_x(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];

    return - a / std::pow(a*x + b, 2.0);
}

__device__ float derivative_psi_t_x(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    // const float b = A[(2*c+1) * n_batch + batch_index];

    return a * exp(t*a);
}

__device__ float derivative_phi_x_x(const float& xini, const float& tini, const float& tm, const int& cm, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index, const float& xmin, const float& xmax, const int& nc){
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dx = 0.0;
    float dthit_dx_dx = 0.0;
    if (cini != cm){
        dthit_dx = derivative_thit_x(xini, cini, tini, A, n_batch, batch_index);
        dthit_dx_dx = derivative_thit_x_x(xini, cini, tini, A, n_batch, batch_index);
    }


    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            xm = xc;
        } 
    }

    float dpsi_dtime = derivative_psi_t(xm, cm, tm, A, n_batch, batch_index);
    float dpsi_dtime_dx = derivative_psi_t_x(xm, cm, tm, A, n_batch, batch_index);
    float dphi_dx = dpsi_dtime_dx * dthit_dx + dpsi_dtime * dthit_dx_dx;
    return dphi_dx;
}

__global__ void kernel_derivative_space_closed_form_dx(
    const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float t, const int xmin, const int xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch){ 
        float result[e];
        integrate_closed_form_trace(result, x[batch_index * n_points + point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
            
        // float phi = result[0];
        float tm = result[1];
        int cm = result[2];
        float dphi_dx_dx = derivative_phi_x_x(x[batch_index * n_points + point_index], t, tm, cm, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
        
        gradpoints[batch_index * n_points + point_index] = dphi_dx_dx;
    }
    return;
}