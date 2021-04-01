#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>

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


__device__ float integrate_closed_form(float x, float t, const float* A,  const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc){

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






__global__ void kernel_get_cell(const int n_points, const float* points, const float xmin, const float xmax, const int nc, int* newpoints){    
    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(point_index < n_points) {
        newpoints[point_index] = get_cell(points[point_index], xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_get_velocity(const int n_points, const int n_batch, 
    const float *points, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = get_velocity(points[point_index], A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_integrate_numeric(const int n_points, const int n_batch, 
    const float *points, const float* A, 
    const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, 
    float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_numeric(points[point_index], t, A, n_batch, batch_index, xmin, xmax, nc, nSteps1, nSteps2);
    }
    return;
}


__global__ void kernel_integrate_closed_form(const int n_points, const int n_batch, 
    const float *points, const float* A, 
    const float xmin, const float xmax, const int nc,  
    float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_closed_form(points[point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}