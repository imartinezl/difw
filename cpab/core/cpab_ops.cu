#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h>

// FUNCTIONS

#define eps FLT_EPSILON;
#define inf INFINITY;

__device__ int sign(const int r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

__device__ int signf(const float r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
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

__device__ float get_psi_DEPRECATED(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    float psi;
    if (cmpf0(a)){
        psi = x + t*b;
    }
    else{
        psi = exp(t*a) * (x + (b/a)) - (b/a);
    }
    return psi;
}

__device__ float get_psi(const float& x, const float& t,  const float& a, const float& b){
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return exp(t*a) * (x + (b/a)) - (b/a);
    }
}

__device__ float get_hit_time_DEPRECATED(const float& x, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
    float xc;
    if( v >= 0.0f ){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    float tcross;
    if (cmpf0(a)){
        tcross = (xc - x)/b;
    }
    else{
        tcross = log((xc + b/a)/(x + b/a))/a;
    }
    return tcross;
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

// NUMERIC
__device__ float get_psi_numeric(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    float psi;
    if (cmpf0(a)){
        psi = x + t*b;
    }
    else{
        psi = std::exp(t*a) * (x + (b/a)) - (b/a);
    }
    return psi;
}


__device__ float get_phi_numeric(const float& x, const float& t, const int& nSteps2, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    // int c;
    for(int j = 0; j < nSteps2; j++) {
        int c = get_cell(x, xmin, xmax, nc);
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, n_batch, batch_index, xmin, xmax, nc);
        c = get_cell(midpoint, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return yn;
}

// INTEGRATION

__device__ float integrate_numeric(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_numeric(xPrev, deltaT, A, n_batch, batch_index, xmin, xmax, nc);
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

__device__ float integrate_closed_form_DEPRECATED(float x, float t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){

    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
        psi = get_psi_DEPRECATED(x, t, A, n_batch, batch_index, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            return psi;
        }
        
        t -= get_hit_time_DEPRECATED(x, A, n_batch, batch_index, xmin, xmax, nc);        
        x = (v >= 0) ? right : left;
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return psi;
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

__device__ void integrate_closed_form_trace_DEPRECATED(float* result, float x, float t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
        psi = get_psi_DEPRECATED(x, t, A, n_batch, batch_index, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            result[0] = psi;
            result[1] = t;
            result[2] = c;
            return;
        }
        
        t -= get_hit_time_DEPRECATED(x, A, n_batch, batch_index, xmin, xmax, nc);        
        x = (v >= 0) ? right : left;
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
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

// DERIVATIVE

__device__ double derivative_psi_theta(const float& x, const float& t, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    double dpsi_dtheta;
    if (cmpf0(a)){
        dpsi_dtheta = t*(x*ak + bk);
    }
    else{
        double tmp = exp(t*a);
        dpsi_dtheta = ak * t * tmp * (x + b/a) + (tmp-1)*(bk*a - ak*b)/pow(a, 2.0);
    }
    return dpsi_dtheta;
}

__device__ double derivative_phi_time(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    double dpsi_dtime;
    if (cmpf0(a)){
        dpsi_dtime = b;
    }
    else{
        dpsi_dtime = exp(t*a)*(a*x + b);
    }
    return dpsi_dtime;
}

__device__ double derivative_thit_theta(const float& x, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    const float v = get_velocity(x, A, n_batch, batch_index, xmin, xmax, nc);
    float xc;
    if( v >= 0){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }

    double dthit_dtheta;
    if (cmpf0(a)){
        dthit_dtheta = (x-xc)*bk / pow(b, 2.0);
    }
    else{
        double d1 = - ak * log( (a*xc + b) / (a*x + b) )/pow(a, 2.0);
        double d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

__device__ double derivative_phi_theta(const float& xini, const float& tm, const int& cm, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    double dthit_dtheta_cum = 0.0;
    if (cini != cm){
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            dthit_dtheta_cum -= derivative_thit_theta(xm, k, d, B, A, n_batch, batch_index, xmin, xmax, nc);
            if (step == 1){
                xm = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xm = left_boundary(c, xmin, xmax, nc);
            }
        }
    }

    const double dpsi_dtheta = derivative_psi_theta(xm, tm, k, d, B, A, n_batch, batch_index, xmin, xmax, nc);
    const double dpsi_dtime = derivative_phi_time(xm, tm, A, n_batch, batch_index, xmin, xmax, nc);
    const double dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}


// OPTIMIZED INTEGRAL

__device__ float get_velocity_optimized_DEPRECATED(const float& x, const int& c, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    return a*x + b;
}

__device__ float get_psi_optimized_DEPRECATED(const float& x, const int&c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return exp(t*a) * (x + (b/a)) - (b/a);
    }
}

__device__ float get_hit_time_optimized_DEPRECATED(const float& x, const int& c, const float& xc, const float* A, const int& n_batch, const int& batch_index){
    const float a = A[(2*c) * n_batch + batch_index];
    const float b = A[(2*c+1) * n_batch + batch_index];
    if (cmpf0(a)){
        return (xc - x)/b;
    }
    else{
        return log((xc + b/a)/(x + b/a))/a;
    }
}

__device__ void integrate_closed_form_trace_optimized_DEPRECATED(float* result, float x, float t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = max(c, nc-1-c);

    float left, right, v, psi, xc;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity_optimized_DEPRECATED(x, c, A, n_batch, batch_index);
        psi = get_psi_optimized_DEPRECATED(x, c, t, A, n_batch, batch_index);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            result[0] = psi;
            result[1] = t;
            result[2] = c;
            return;
        }
        
        xc = (v >= 0) ? right : left;
        t -= get_hit_time_optimized_DEPRECATED(x, c, xc, A, n_batch, batch_index);      
        x = xc;  
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
}

// OPTIMIZED NUMERIC

__device__ float get_psi_numeric_optimized(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
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

__device__ float get_phi_numeric_optimized(const float& x, const float& t, const int& nSteps2, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    for(int j = 0; j < nSteps2; j++) {
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, n_batch, batch_index, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return yn;
}

__device__ float integrate_numeric_optimized(const float& x, const float& t, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_numeric_optimized(xPrev, c, deltaT, A, n_batch, batch_index);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric_optimized(xPrev, deltaT, nSteps2, A, n_batch, batch_index, xmin, xmax, nc);
            c = get_cell(xPrev, xmin, xmax, nc);
        }
    }
    return xPrev;
}

// OPTIMIZED DERIVATIVE

__device__ double derivative_psi_theta_optimized_DEPRECATED(const double& x, const int& c, const float& t, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    if (cmpf0(a)){
        return t*(x*ak + bk);
    }
    else{
        double tmp = exp(t*a);
        return ak * t * tmp * (x + b/a) + (tmp-1)*(bk*a - ak*b)/pow(a, 2.0);
    }
}

__device__ void derivative_psi_theta_optimized(double* gradpoints, const float& x, const int& c, const float& t, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index){
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

__device__ float derivative_phi_time_optimized(const float& x, const int& c, const float& t, const float* A, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    if (cmpf0(a)){
        return b;
    }
    else{
        return exp(t*a)*(a*x + b);
    }
}

__device__ float derivative_thit_theta_optimized_DEPRECATED(const double& x, const int& c, const float& xc, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index){
    const double a = A[(2*c) * n_batch + batch_index];
    const double b = A[(2*c+1) * n_batch + batch_index];

    const double ak = B[(2*c)*d + k];
    const double bk = B[(2*c+1)*d + k];

    double dthit_dtheta;
    if (cmpf0(a)){
        dthit_dtheta = (x-xc)*bk / pow(b, 2.0);
    }
    else{
        double d1 = - ak * log( (a*xc + b) / (a*x + b) )/pow(a, 2.0);
        double d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

__device__ void derivative_thit_theta_optimized(double* gradpoints, const float& x, const int& c, const float& xc, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index){
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

__device__ float derivative_phi_theta_optimized_DEPRECATED(const float& xini, const float& tm, const int& cm, const int& k, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    double dthit_dtheta_cum = 0.0;
    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            dthit_dtheta_cum -= derivative_thit_theta_optimized_DEPRECATED(xm, c, xc, k, d, B, A, n_batch, batch_index);
            xm = xc;
        }
    }

    const double dpsi_dtheta = derivative_psi_theta_optimized_DEPRECATED(xm, cm, tm, k, d, B, A, n_batch, batch_index);
    const double dpsi_dtime = derivative_phi_time_optimized(xm, cm, tm, A, n_batch, batch_index);
    const double dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}

__device__ void derivative_phi_theta_optimized(double* gradpoints, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const int& n_batch, const int& batch_index, const int& n_points, const int& point_index, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    // float dthit_dtheta_cum[d] = { };
    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            derivative_thit_theta_optimized(gradpoints, xm, c, xc, d, B, A, n_batch, batch_index, n_points, point_index);
            xm = xc;
        }
    }

    // TODO: OPTIONAL change name derivative_phi_time => derivative_phi_thit NO!!! better dthit_dtheta => dtime_dtheta
    const float dpsi_dtime = derivative_phi_time_optimized(xm, cm, tm, A, n_batch, batch_index);
    for(int k=0; k < d; k++){
        gradpoints[batch_index*(n_points * d) + point_index*d + k] *= dpsi_dtime;
    }
    // float dpsi_dtheta[d] = { };
    derivative_psi_theta_optimized(gradpoints, xm, cm, tm, d, B, A, n_batch, batch_index, n_points, point_index);
    
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
        newpoints[batch_index * n_points + point_index] = get_velocity(x[point_index], A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_integrate_numeric(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, 
    const int nSteps1, const int nSteps2, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_numeric_optimized(x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc, nSteps1, nSteps2);
    }
    return;
}

__global__ void kernel_integrate_closed_form(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    float t = 1.0;
    if(point_index < n_points && batch_index < n_batch) {
        newpoints[batch_index * n_points + point_index] = integrate_closed_form(x[point_index], t, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_derivative_closed_form(
    const int n_points, const int n_batch, const int d,
    const float* x, const float* A, const float* B, 
    const int xmin, const int xmax, const int nc, double* gradpoints){

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

__global__ void kernel_integrate_closed_form_trace(
    const int n_points, const int n_batch, const float* x, const float* A, 
    const float xmin, const float xmax, const int nc, float* newpoints){

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
    const float xmin, const float xmax, const int nc, double* gradpoints){

    int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_index = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_index = blockIdx.z * blockDim.z + threadIdx.z;
    
    // float t = 1.0;
    const int e = 3;

    if(point_index < n_points && batch_index < n_batch && dim_index < d) {
        // float phi = newpoints[batch_index*(n_points * e) + point_index*e + 0];
        float tm = newpoints[batch_index*(n_points * e) + point_index*e + 1];
        int cm = newpoints[batch_index*(n_points * e) + point_index*e + 2];
        gradpoints[batch_index*(n_points * d) + point_index*d + dim_index] = derivative_phi_theta_optimized_DEPRECATED(x[point_index], tm, cm, dim_index, d, B, A, n_batch, batch_index, xmin, xmax, nc);
    }
    return;
}

__global__ void kernel_derivative_closed_form_trace_optimized(
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
        
        derivative_phi_theta_optimized(gradpoints, x[point_index], tm, cm, d, B, A, n_batch, batch_index, n_points, point_index, xmin, xmax, nc);
        // for(int k = 0; k < d; k++){ // for all parameters theta
        //     gradpoints[batch_index*(n_points * d) + point_index*d + k] = dphi_dtheta[k];
        // }
    }
    return;
}