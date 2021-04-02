#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>

// FUNCTIONS

__device__ int sign(int r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

__device__ bool cmpf(float x, float y, float eps = 1e-6f){
    // return x == y;
    return fabs(x - y) < eps;
}

__device__ float right_boundary(int c, const float xmin, const float xmax, const int nc){
    // float eps = std::numeric_limits<float>::epsilon();
    // eps = 1e-5f;
    float eps = FLT_EPSILON;
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

__device__ float left_boundary(int c, const float xmin, const float xmax, const int nc){
    // float eps = std::numeric_limits<float>::epsilon();
    // eps = 1e-5f;
    float eps = FLT_EPSILON;
    return xmin + c * (xmax - xmin) / nc - eps;
}

__device__ int get_cell(float x, const float xmin, const float xmax, const int nc){
    int c = floor((x - xmin) / (xmax - xmin) * nc);
    c = max(0, min(c, nc-1));
    return c;
}

__device__ float get_velocity(float x, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];
    return a*x + b;
}

__device__ float get_psi(float x, float t, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];
    float psi;
    if (cmpf(a, 0.0f)){
        psi = x + t*b;
    }
    else{
        psi = exp(t*a) * (x + (b/a)) - (b/a);
    }
    return psi;
}

__device__ float get_hit_time(float x, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
    float xc;
    if( v >= 0.0f ){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];
    float tcross;
    if (cmpf(a, 0.0f)){
        tcross = (xc - x)/b;
    }
    else{
        tcross = log((xc + b/a)/(x + b/a))/a;
    }
    return tcross;
}


__device__ float get_phi_numeric(float x, float t, int nSteps2, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    float yn = x;
    float midpoint;
    float deltaT = t / nSteps2;
    // int c;
    for(int j = 0; j < nSteps2; j++) {
        int c = get_cell(x, xmin, xmax, nc);
        midpoint = yn + deltaT / 2 * get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
        c = get_cell(midpoint, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return yn;
}

// INTEGRATION

__device__ float integrate_numeric(float x, float t, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2){
    float xPrev = x;
    float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi(xPrev, deltaT, A, n_batch, batch_index, xmin, xmax, nc);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        float xNum = get_phi_numeric(xPrev, deltaT, nSteps2, A, n_batch, batch_index, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = xNum;
        }
        c = get_cell(xPrev, xmin, xmax, nc);
    }
    return xPrev;
}


__device__ float integrate_closed_form(float x, float t, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){

    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    int contmax = max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
        psi = get_psi(x, t, A, n_batch, batch_index, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            return psi;
        }
        
        t -= get_hit_time(x, A, n_batch, batch_index, xmin, xmax, nc);        
        x = (v >= 0) ? right : left;
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return psi;
}

// DERIVATIVE

__device__ float derivative_psi_theta(float x, float t, const int k, const int d, const float* B, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];

    // float ak = B[2*nc*k + 2*c];
    // float bk = B[2*nc*k + 2*c + 1];
    float ak = B[(2*c)*d + k];
    float bk = B[(2*c+1)*d + k];

    float dpsi_dtheta;
    if (cmpf(a, 0.0f)){
        dpsi_dtheta = t*(x*ak + bk);
    }
    else{
        dpsi_dtheta = ak * t * exp(a*t) * (x + b/a) + (exp(t*a)-1)*(bk*a - ak*b)/pow(a, 2.0);
    }
    return dpsi_dtheta;
}

__device__ float derivative_phi_time(float x, float t, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];

    float dpsi_dtime;
    if (cmpf(a, 0.0f)){
        dpsi_dtime = b;
    }
    else{
        dpsi_dtime = exp(t*a)*(a*x + b);
    }
    return dpsi_dtime;
}

__device__ float derivative_thit_theta(float x, const int k, const int d, const float* B, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[(2*c) * n_batch + batch_index];
    float b = A[(2*c+1) * n_batch + batch_index];

    // float ak = B[2*nc*k + 2*c];
    // float bk = B[2*nc*k + 2*c + 1];
    float ak = B[(2*c)*d + k];
    float bk = B[(2*c+1)*d + k];

    float v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
    float xc;
    if( v >= 0){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }

    float dthit_dtheta;
    if (cmpf(a, 0.0f)){
        dthit_dtheta = (x-xc)*bk / pow(b, 2.0);
    }
    else{
        float d1 = - ak * log( (a*xc + b) / (a*x + b) )/pow(a, 2.0);
        float d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

__device__ float derivative_phi_theta(float xini, float tm, int cm, const int k, const int d, const float* B, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    
    int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dtheta_cum = 0.0;
    if (cini != cm){
        int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            dthit_dtheta_cum -= derivative_thit_theta(xm, k, d, B, A, n_batch, batch_index, xmin, xmax, nc);
            if (step == 1){
                xm = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xm = left_boundary(c, xmin, xmax, nc);
            }
        }
    }

    float dpsi_dtheta = derivative_psi_theta(xm, tm, k, d, B, A, n_batch, batch_index, xmin, xmax, nc);
    float dpsi_dtime = derivative_phi_time(xm, tm, A, n_batch, batch_index, xmin, xmax, nc);
    float dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}

__device__ void integrate_closed_form_trace(float* result, float x, float t, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    int contmax = max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
        psi = get_psi(x, t, A, n_batch, batch_index, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            result[0] = psi;
            result[1] = t;
            result[2] = c;
            return;
        }
        
        t -= get_hit_time(x, A, n_batch, batch_index, xmin, xmax, nc);        
        x = (v >= 0) ? right : left;
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
}




__global__ void kernel_get_cell(const int n_points, const float* points, const float xmin, const float xmax, const int nc, int* newpoints){    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(point_index < n_points) {
        newpoints[point_index] = get_cell(points[point_index], xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_get_velocity(const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = get_velocity(x[point_index], A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_integrate_numeric(const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, 
    float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_numeric(x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc, nSteps1, nSteps2);
    }
    return;
}


__global__ void kernel_integrate_closed_form(const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc,  
    float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_closed_form(x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_derivative_closed_form(const int n_points, const int n_batch, const int d,
    const float* x, const float* A, const float* B, const int xmin, const int xmax, const int nc, float* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    // int dim_index = blockIdx.z * blockDim.z + threadIdx.z;
    
    float t = 1.0;
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch){ // && dim_index < d){
        float result[e];
        integrate_closed_form_trace(result, x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
        for(int k = 0; k < d; k++){
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];
            gradpoints[batch_index*(n_points * d) + point_index*d + k] = derivative_phi_theta(x[point_index], tm, cm, k, d, B, A, n_batch, batch_index, xmin, xmax, nc);
        }
    }
    return;
}



__global__ void kernel_integrate_closed_form_trace(const int n_points, const int n_batch, 
    const float* x, const float* A, 
    const float xmin, const float xmax, const int nc,  
    float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    
    float t = 1.0;
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch) {
        float result[e];
        integrate_closed_form_trace(result, x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
        for(int p = 0; p < e; p++){
            newpoints[batch_index*(n_points * e) + point_index*e + p] = result[p];
        }
    }
    return;
}


__global__ void kernel_derivative_closed_form_trace(
    const int n_points, const int n_batch, const int d,
    const float* newpoints, const float* x, const float* A, const float* B, 
    const float xmin, const float xmax, const int nc, 
    float* gradpoints){
        int point_index = blockIdx.x * blockDim.x + threadIdx.x;
        int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
        int dim_index = blockIdx.z * blockDim.z + threadIdx.z;
        
        // float t = 1.0;
        const int e = 3;
    
        if(point_index < n_points && batch_index < n_batch && dim_index < d) {
            // float phi = newpoints[batch_index*(n_points * e) + point_index*e + 0];
            float tm = newpoints[batch_index*(n_points * e) + point_index*e + 1];
            int cm = newpoints[batch_index*(n_points * e) + point_index*e + 2];
            gradpoints[batch_index*(n_points * d) + point_index*d + dim_index] = derivative_phi_theta(x[point_index], tm, cm, dim_index, d, B, A, n_batch, batch_index, xmin, xmax, nc);
        }
        return;
    }