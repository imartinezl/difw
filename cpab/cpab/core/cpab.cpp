// #include <torch/extension.h>
// #include "../core/cpab_ops.h"

#include <math.h>
#include <iostream>
#include <vector>
#include <limits>


int sign(int r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

// TODO: replace 2 for params per cell
bool cmpf(float x, float y, float eps = 1e-6f)
{
    // return x == y;
    return std::fabs(x - y) < eps;
}

int get_cell(float x, const float xmin, const float xmax, const int nc){
    int c = std::floor((x - xmin) / (xmax - xmin) * nc);
    c = std::max(0, std::min(c, nc-1));
    return c;
}

float right_boundary(int c, const float xmin, const float xmax, const int nc){
    float eps = std::numeric_limits<float>::epsilon();
    // eps = 1e-5f;
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

float left_boundary(int c, const float xmin, const float xmax, const int nc){
    float eps = std::numeric_limits<float>::epsilon();
    // eps = 1e-5f;
    return xmin + c * (xmax - xmin) / nc - eps;
}

float get_velocity(float x, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[2*c];
    float b = A[2*c+1];
    return a*x + b;
}

float get_psi(float x, float t, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[2*c];
    float b = A[2*c+1];
    float psi;
    // if (a == 0.0){
    if (cmpf(a, 0.0f)){
        psi = x + t*b;
    }
    else{
        psi = std::exp(t*a) * (x + (b/a)) - (b/a);
    }
    return psi;
}

float get_hit_time(float x, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float v = get_velocity(x, A, xmin, xmax, nc);
    float xc;
    if( v >= 0.0f ){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }
    float a = A[2*c];
    float b = A[2*c+1];
    float tcross;
    // if (a == 0.0){
    if (cmpf(a, 0.0f)){
        tcross = (xc - x)/b;
    }
    else{
        tcross = std::log((xc + b/a)/(x + b/a))/a;
    }
    return tcross;
}


// INTEGRATION

float integrate_closed_form(float x, float t, const float* A, const float xmin, const float xmax, const int nc){

    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    int contmax = std::max(c, nc-1-c);

    float left, right, v, psi, thit;
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

float get_numeric_phi(float x, float t, int nSteps2, const float* A, const float xmin, const float xmax, const int nc){
    float yn = x;
    float midpoint;
    float deltaT = t / nSteps2;
    int c;
    for(int j = 0; j < nSteps2; j++) {
        c = get_cell(x, xmin, xmax, nc);
        midpoint = yn + deltaT / 2 * get_velocity(x, A, xmin, xmax, nc);
        c = get_cell(midpoint, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, xmin, xmax, nc);
    }
    return yn;
}

float integrate_numeric(float x, float t, const float* A, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2){
    float xPrev = x;
    float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi(xPrev, deltaT, A, xmin, xmax, nc);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        float xNum = get_numeric_phi(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
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


// DERIVATIVE

// TODO: remove method
float integrate_closed_form_trace_full(float x, float t, const float* A, const float xmin, const float xmax, const int nc, std::vector<float> &xr, std::vector<float> &tr){
    int cont = 0;

    xr.push_back(x);
    tr.push_back(t);

    int c;
    float left, right, psi, thit;
    while (true) {
        c = get_cell(x, xmin, xmax, nc);
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        psi = get_psi(x, t, A, xmin, xmax, nc);

        if ((left <= psi) && (psi <= right)){
            return psi;
        }
        else{
            thit = get_hit_time(x, A, xmin, xmax, nc);
            t -= thit;
        }

        if (psi < left){
            x = left;
        }else if (psi > right){
            x = right;
        }
        xr.push_back(x);
        tr.push_back(t);

        cont++;
        if (cont > nc){
            break;
        }
    }
}

void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    int contmax = std::max(c, nc-1-c);

    float left, right, v, psi, thit;
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


float derivative_psi_theta(float x, float t, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[2*c];
    float b = A[2*c + 1];

    float ak = B[2*nc*k + 2*c];
    float bk = B[2*nc*k + 2*c + 1];

    float dpsi_dtheta;
    // if (a == 0.0){
    if (cmpf(a, 0.0f)){
        dpsi_dtheta = t*(x*ak + bk);
    }
    else{
        dpsi_dtheta = ak * t * std::exp(a*t) * (x + b/a) + (std::exp(t*a)-1)*(bk*a - ak*b)/std::pow(a, 2.0);
    }
    return dpsi_dtheta;
}

float derivative_phi_time(float x, float t, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[2*c];
    float b = A[2*c + 1];

    float dpsi_dtime;
    // if (a == 0.0){
    if (cmpf(a, 0.0f)){
        dpsi_dtime = b;
    }
    else{
        dpsi_dtime = std::exp(t*a)*(a*x + b);
    }
    return dpsi_dtime;
}

float derivative_thit_theta(float x, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    int c = get_cell(x, xmin, xmax, nc);
    float a = A[2*c];
    float b = A[2*c + 1];

    float ak = B[2*nc*k + 2*c];
    float bk = B[2*nc*k + 2*c + 1];

    float v = get_velocity(x, A, xmin, xmax, nc);
    float xc;
    if( v >= 0){
        xc = right_boundary(c, xmin, xmax, nc);
    }
    else{
        xc = left_boundary(c, xmin, xmax, nc);
    }

    float dthit_dtheta;
    // if (a == 0.0){
    if (cmpf(a, 0.0f)){
        dthit_dtheta = (x-xc)*bk / std::pow(b, 2.0);
    }
    else{
        float d1 = - ak * std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        float d2 = (x - xc) * ( bk*a - ak*b) / (a * (a*x + b) * (a*xc + b) );
        dthit_dtheta = d1 + d2;
    }
    return dthit_dtheta;
}

// TODO: remove method
float derivative_phi_theta_old(std::vector<float> &xr, std::vector<float> &tr, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    
    float dthit_dtheta_cum = 0.0;
    int iters = xr.size();
    for (int i = 0; i < (iters-1); i++) {
        dthit_dtheta_cum -= derivative_thit_theta(xr[i], k, B, A, xmin, xmax, nc);
    }

    float x = xr[iters-1];
    float t = tr[iters-1];

    float dpsi_dtheta = derivative_psi_theta(x, t, k, B, A, xmin, xmax, nc);
    float dpsi_dtime = derivative_phi_time(x, t, A, xmin, xmax, nc);
    float dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}


float derivative_phi_theta(float xini, float tm, int cm, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    
    int cini = get_cell(xini, xmin, xmax, nc);
    float xm = xini;

    float dthit_dtheta_cum = 0.0;
    if (cini != cm){
        int step = sign(cm - cini);
        for (int c = cini; step*c < cm*step; c += step){
            dthit_dtheta_cum -= derivative_thit_theta(xm, k, B, A, xmin, xmax, nc);
            if (step == 1){
                xm = right_boundary(c, xmin, xmax, nc);
            }else if (step == -1){
                xm = left_boundary(c, xmin, xmax, nc);
            }
        }
        
    }

    float dpsi_dtheta = derivative_psi_theta(xm, tm, k, B, A, xmin, xmax, nc);
    float dpsi_dtime = derivative_phi_time(xm, tm, A, xmin, xmax, nc);
    float dphi_dtheta = dpsi_dtheta + dpsi_dtime*dthit_dtheta_cum;    

    return dphi_dtheta;
}
