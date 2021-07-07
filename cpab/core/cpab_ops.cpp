#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

// FUNCTIONS
float eps = std::numeric_limits<float>::epsilon();
float inf = std::numeric_limits<float>::infinity();

int sign(const int r){
    return (r > 0) - (r < 0);
}

int signf(const float r){
    return (r > 0) - (r < 0);
}

// TODO: replace 2 for params per cell
bool cmpf(float x, float y)
{
    return std::fabs(x - y) < eps;
}

bool cmpf0(const float& x)
{   
    return std::fabs(x) < eps;
}

float right_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
    return xmin + (c + 1) * (xmax - xmin) / nc + eps;
}

float left_boundary(const int& c, const float& xmin, const float& xmax, const int& nc){
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

// INTEGRATION CLOSED FORM

float get_psi(const float& x, const float& t, const float& a, const float& b){
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        const float eta = std::exp(t*a);
        return eta * x + (b/a) * (eta - 1.0);
        // return std::exp(t*a) * (x + (b/a)) - (b/a);
    }
}

float get_hit_time(float x, int c, const float& a, const float& b, const float& xmin, const float& xmax, const int& nc, float& xc, int& cc){

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

float integrate_closed_form(float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[2*c];
        b = A[2*c+1];

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

void integrate_closed_form_trace(float* result, float x, float t, const float* A, const float& xmin, const float& xmax, const int& nc){
    int c = get_cell(x, xmin, xmax, nc);
    int cont = 0;
    const int contmax = std::max(c, nc-1-c);

    float a, b, xc, thit;
    int cc;
    while (true) {
        a = A[2*c];
        b = A[2*c+1];

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

float get_psi_numeric(const float& x, const int& c, const float& t, const float* A){
    const float a = A[2*c];
    const float b = A[2*c+1];
    if (cmpf0(a)){
        return x + t*b;
    }
    else{
        return std::exp(t*a) * (x + (b/a)) - (b/a);
    }
}

float get_phi_numeric(const float& x, const float& t, const int& nSteps2, const float* A, const float& xmin, const float& xmax, const int& nc){
    float yn = x;
    float midpoint;
    const float deltaT = t / nSteps2;
    for(int j = 0; j < nSteps2; j++) {
        midpoint = yn + deltaT / 2 * get_velocity(yn, A, xmin, xmax, nc);
        yn = yn + deltaT * get_velocity(midpoint, A, xmin, xmax, nc);
    }
    return yn;
}

float integrate_numeric(const float& x, const float& t, const float* A, const float& xmin, const float& xmax, const int& nc, const int& nSteps1, const int& nSteps2){
    float xPrev = x;
    const float deltaT = t / nSteps1;
    int c = get_cell(xPrev, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi_numeric(xPrev, c, deltaT, A);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        if (c == cTemp){
            xPrev = xTemp;
        }
        else{
            xPrev = get_phi_numeric(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
            c = get_cell(xPrev, xmin, xmax, nc);
        }
    }
    return xPrev;
}



// DERIVATIVE

void derivative_psi_theta(float* dpsi_dtheta, const float& x, const int& c, const float& t, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const double a = A[2*c];
    const double b = A[2*c + 1];


    if (cmpf0(a)){
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = t*(x*ak + bk);
        }
    }
    else{
        const double tmp = std::exp(t*a);
        const double tmp1 = t * tmp * (x + b/a);
        const double tmp2 = (tmp-1)/std::pow(a, 2.0);
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];
            dpsi_dtheta[k] = ak * tmp1 + tmp2 * (bk*a - ak*b);
        }
        // return = ak * t * std::exp(t*a) * (x + b/a) + (std::exp(t*a)-1)*(bk*a - ak*b)/std::pow(a, 2.0);

    }
}

float derivative_phi_time(const float& x, const int& c, const float& t, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const float a = A[2*c];
    const float b = A[2*c + 1];

    if (cmpf0(a)){
        return b;
    }
    else{
        return std::exp(t*a)*(a*x + b);
    }
}

void derivative_thit_theta(float* dthit_dtheta_cum, const float& x, const int& c, const float& xc, const int& d, const float* B, const float* A){
    // int c = get_cell(x, xmin, xmax, nc);
    const double a = A[2*c];
    const double b = A[2*c + 1];

    if (cmpf0(a)){
        const double tmp = (x-xc) / std::pow(b, 2.0);
        for(int k=0; k < d; k++){
            const double bk = B[(2*c+1)*d + k];
            dthit_dtheta_cum[k] -= tmp*bk;
        }
    }
    else{
        const double tmp1 = std::log( (a*xc + b) / (a*x + b) )/std::pow(a, 2.0);
        const double tmp2 = (x - xc) / (a * (a*x + b) * (a*xc + b) );
        for(int k=0; k < d; k++){
            const double ak = B[(2*c)*d + k];
            const double bk = B[(2*c+1)*d + k];

            const double d1 = - ak * tmp1;
            const double d2 = ( bk*a - ak*b) * tmp2;
            dthit_dtheta_cum[k] -= d1 + d2; 
        }
        return;
    }
}

void derivative_phi_theta(float* dphi_dtheta, const float& xini, const float& tm, const int& cm, const int& d, const float* B, const float* A, const float& xmin, const float& xmax, const int& nc){ 
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
            derivative_thit_theta(dthit_dtheta_cum, xm, c, xc, d, B, A);
            xm = xc;
        } 
    }

    const float dpsi_dtime = derivative_phi_time(xm, cm, tm, A);
    float dpsi_dtheta[d] = { };
    derivative_psi_theta(dpsi_dtheta, xm, cm, tm, d, B, A);
    for(int k=0; k < d; k++){
        dphi_dtheta[k] = dpsi_dtheta[k] + dpsi_dtime*dthit_dtheta_cum[k];    
    }
}