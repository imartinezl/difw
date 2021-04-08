// #include <math.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

// FUNCTIONS

int sign(const int r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

// TODO: replace 2 for params per cell
bool cmpf(float x, float y, float eps = 1e-6f)
{   
    return std::fabs(x - y) < eps;
}

bool cmpf0(const float& x, float eps = 1e-6f)
{   
    // eps = 1e-6f;
    return std::fabs(x) < eps;
}

float eps = std::numeric_limits<float>::epsilon();

float right_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    // eps = 1e-5f;
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

float left_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    // eps = 1e-5f;
    return xmin + c * (xmax - xmin) / nc - eps;
}

int get_cell(const float& x, const float& xmin, const float& xmax, const int& nc){
    int c = std::floor((x - xmin) / (xmax - xmin) * nc);
    c = std::max(0, std::min(c, nc-1));
    return c;
}

float get_velocity(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c+1];
    return a*x + b;
}

float get_psi(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c+1];
    float psi;
    if (cmpf0(a)){
        psi = x + t*b;
    }
    else{
        psi = std::exp(t*a) * (x + (b/a)) - (b/a);
    }
    return psi;
}

float get_hit_time(const float& x, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float v = get_velocity(x, A, xmin, xmax, nc);
    float xc;
    if( v >= 0.0f ){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }
    const float a = A[2*c];
    const float b = A[2*c+1];
    float tcross;
    if (cmpf0(a)){
        tcross = (xc - x)/b;
    }
    else{
        tcross = std::log((xc + b/a)/(x + b/a))/a;
    }
    return tcross;
}


// NUMERIC
float get_phi_numeric(const float& x, const float& t, const int& nSteps2, const float* A, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    int c;
    for(int j = 0; j < nSteps2; j++) {
        c = get_cell(x, xmin, xmax, nc);
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, xmin, xmax, nc);
        c = get_cell(midpoint, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, xmin, xmax, nc);
    }
    return yn;
}

// INTEGRATION

float integrate_numeric(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    for(int j = 0; j < nSteps1; j++) {
        int c = get_cell(xPrev, xmin, xmax, nc);
        float xTemp = get_psi(xPrev, deltaT, A, xmin, xmax, nc);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
        }
    }
    return xPrev;
}

float integrate_closed_form(float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){

    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, xmin, xmax, nc);
        psi = get_psi(x, t, A, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            return psi;
        }
        
        t -= get_hit_time(x, A, xmin, xmax, nc);        
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

float derivative_psi_theta(const float& x, const float& t, const int& k, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const float ak = B[(2*c)*d + k];
    const float bk = B[(2*c+1)*d + k];

    float dpsi_dtheta;
    if (cmpf0(a)){
        dpsi_dtheta = t*(x*ak + bk);
    }
    else{
        float tmp = std::exp(t*a);
        dpsi_dtheta = ak * t * tmp * (x + b/a) + (tmp-1)*(bk*a - ak*b)/std::pow(a, 2.0);
    }
    return dpsi_dtheta;
}

float derivative_phi_time(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    float dpsi_dtime;
    if (cmpf0(a)){
        dpsi_dtime = b;
    }
    else{
        dpsi_dtime = std::exp(t*a)*(a*x + b);
    }
    return dpsi_dtime;
}

float derivative_thit_theta(const float& x, const int& k, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){
    const int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const float ak = B[(2*c)*d + k];
    const float bk = B[(2*c+1)*d + k];

    const float v = get_velocity(x, A, xmin, xmax, nc);
    float xc;
    if( v >= 0){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }

    float dthit_dtheta;
    if (cmpf0(a)){
        dthit_dtheta = (x-xc)*bk / std::pow(b, 2.0);
    }
    else{
        float d1 = - ak * std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        float d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

float derivative_phi_theta(const float& xini, const float& tm, const int& cm, const int& k, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dtheta_cum = 0.0;
    if (cini != cm){
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            dthit_dtheta_cum -= derivative_thit_theta(xm, k, d, B, A, xmin, xmax, nc);
            if (step == 1){
                xm = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xm = left_boundary(c, xmin, xmax, nc);
            }
        } 
    }

    const float dpsi_dtheta = derivative_psi_theta(xm, tm, k, d, B, A, xmin, xmax, nc);
    const float dpsi_dtime = derivative_phi_time(xm, tm, A, xmin, xmax, nc);
    const float dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;

    return dphi_dtheta;
}


// TRANSFORMATION

void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float left, right, v, psi;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity(x, A, xmin, xmax, nc);
        psi = get_psi(x, t, A, xmin, xmax, nc);

        cond1 = (left <= psi) && (psi <= right);
        cond2 = (v >= 0) && (c == nc-1);
        cond3 = (v <= 0) && (c == 0);

        if (cond1 || cond2 || cond3){
            result[0] = psi;
            result[1] = t;
            result[2] = c;
            return;
        }
        
        t -= get_hit_time(x, A, xmin, xmax, nc);        
        x = (v >= 0) ? right : left;
        c = (v >= 0) ? c+1 : c-1;

        cont++;
        if (cont > contmax){
            break;
        }
    }
    return;
}


// OPTIMIZED INTEGRAL

float get_velocity_optimized(const float& x, const int& c, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c+1];
    return a*x + b;
}

float get_psi_optimized(const float& x, const int& c, const float& t, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c+1];
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return std::exp(t*a) * (x + (b/a)) - (b/a);
    }
}

float get_hit_time_optimized(const float& x, const int& c, const float& xc, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    // float v = get_velocity(x, A, xmin, xmax, nc);
    // float xc;
    // if( v >= 0.0f ){
    //     xc = right_boundary(c, xmin, xmax, nc);
    // }
    // else{
    //     xc = left_boundary(c, xmin, xmax, nc);
    // }
    const float a = A[2*c];
    const float b = A[2*c+1];
    if (cmpf0(a)){
        return (xc - x)/b;
    }
    else{
        return std::log((xc + b/a)/(x + b/a))/a;
    }
}

void integrate_closed_form_trace_optimized(float* result, float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float left, right, v, psi, xc;
    bool cond1, cond2, cond3;
    while (true) {
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        v = get_velocity_optimized(x, c, A);
        psi = get_psi_optimized(x, c, t, A);

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
        t -= get_hit_time_optimized(x, c, xc, A);
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

float get_phi_numeric_optimized(const float& x, const float& t, const int& nSteps2, const float* A, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    // int c;
    for(int j = 0; j < nSteps2; j++) {
        // c = get_cell(x, xmin, xmax, nc);
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, xmin, xmax, nc);
        // c = get_cell(midpoint, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, xmin, xmax, nc);
    }
    return yn;
}

float integrate_numeric_optimized(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(xPrev, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_optimized(xPrev, c, deltaT, A);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric_optimized(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
            c = get_cell(xPrev, xmin, xmax, nc);
        }
    }
    return xPrev;
}


// OPTIMIZED DERIVATIVE

// TODO: to be removed, it is here to improve the understanding of these operations
float derivative_psi_theta_optimized_old(const float& x, const int& c, const float& t, const int& k, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const float ak = B[(2*c)*d + k];
    const float bk = B[(2*c+1)*d + k];

    if (cmpf0(a)){
        return t*(x*ak + bk);
    }
    else{
        float tmp = std::exp(t*a);
        return ak * t * tmp * (x + b/a) + (tmp-1)*(bk*a - ak*b)/std::pow(a, 2.0);
        // return = ak * t * std::exp(t*a) * (x + b/a) + (std::exp(t*a)-1)*(bk*a - ak*b)/std::pow(a, 2.0);
    }
}

void derivative_psi_theta_optimized(float* dpsi_dtheta, const float& x, const int& c, const float& t, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];


    if (cmpf0(a)){
        for(int k=0; k < d; k++){
            const float ak = B[(2*c)*d + k];
            const float bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = t*(x*ak + bk);
        }
    }
    else{
        const float tmp = std::exp(t*a);
        const float tmp1 = t * tmp * (x + b/a);
        const float tmp2 = (tmp-1)/std::pow(a, 2.0);
        for(int k=0; k < d; k++){
            const float ak = B[(2*c)*d + k];
            const float bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = ak * tmp1 + tmp2 * (bk*a - ak*b);
        }
        // return = ak * t * std::exp(t*a) * (x + b/a) + (std::exp(t*a)-1)*(bk*a - ak*b)/std::pow(a, 2.0);
    }
}

float derivative_phi_time_optimized(const float& x, const int& c, const float& t, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    // float dpsi_dtime;
    // if (a == 0.0){
    if (cmpf0(a)){
        return b;
    }
    else{
        return std::exp(t*a)*(a*x + b);
    }
    // return dpsi_dtime;
}

// TODO: to be removed, it is here to improve the understanding of these operations
float derivative_thit_theta_optimized_old(const float& x, const int& c, const float& xc, const int& k, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    const float ak = B[(2*c)*d + k];
    const float bk = B[(2*c+1)*d + k];

    // float v = get_velocity_optimized(x, c, A);
    // float xc;
    // if( v >= 0){
    //     xc = right_boundary(c, xmin, xmax, nc);
    // }
    // else{
    //     xc = left_boundary(c, xmin, xmax, nc);
    // }

    float dthit_dtheta;
    // if (a == 0.0){
    if (cmpf0(a)){
        dthit_dtheta = (x-xc)*bk / std::pow(b, 2.0);
    }
    else{
        const float d1 = - ak * std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        const float d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

void derivative_thit_theta_optimized(float* dthit_dtheta_cum, const float& x, const int& c, const float& xc, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    if (cmpf0(a)){
        const float tmp = (x-xc) / std::pow(b, 2.0);
        for(int k=0; k < d; k++){
            const float bk = B[(2*c+1)*d + k];
            dthit_dtheta_cum[k] -= tmp*bk;
        }
    }
    else{
        const float tmp1 = std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        const float tmp2 = (x - xc) / (a * (a*x + b) * (a*xc + b) );
        for(int k=0; k < d; k++){
            const float ak = B[(2*c)*d + k];
            const float bk = B[(2*c+1)*d + k];

            const float d1 = - ak * tmp1;
            const float d2 = ( bk*a - ak*b) * tmp2;;
            dthit_dtheta_cum[k] -= d1 + d2;
        }
    }
}

// TODO: to be removed, it is here to improve the understanding of these operations
float derivative_phi_theta_optimized_old(const float& xini, const float& tm, const int& cm, const int& k, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){
    
    const int cini = get_cell(xini, xmin, xmax, nc);
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
            dthit_dtheta_cum -= derivative_thit_theta_optimized_old(xm, c, xc, k, d, B, A);
            xm = xc;
        } 
    }

    const float dpsi_dtheta = derivative_psi_theta_optimized_old(xm, cm, tm, k, d, B, A);
    const float dpsi_dtime = derivative_phi_time_optimized(xm, cm, tm, A);
    const float dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}

void derivative_phi_theta_optimized(float* dphi_dtheta, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){ 
    const int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dtheta_cum[d] = { };
    if (cini != cm){
        float xc;
        const int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            if (step == 1){
                xc = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xc = left_boundary(c, xmin, xmax, nc);
            }
            derivative_thit_theta_optimized(dthit_dtheta_cum, xm, c, xc, d, B, A);
            xm = xc;
        } 
    }

    // float dphi_dtheta[d] = { };
    const float dpsi_dtime = derivative_phi_time_optimized(xm, cm, tm, A);
    float dpsi_dtheta[d] = { };
    derivative_psi_theta_optimized(dpsi_dtheta, xm, cm, tm, d, B, A);
    for(int k=0; k < d; k++){
        // const float dpsi_dtheta = derivative_psi_theta_optimized(xm, cm, tm, k, d, B, A);
        dphi_dtheta[k] = dpsi_dtheta[k] + dpsi_dtime*dthit_dtheta_cum[k];    
    }

    // return dphi_dtheta;
}