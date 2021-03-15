#include <torch/extension.h>
// #include "../core/cpab_ops.h"
#include <iostream>


// TODO: replace 2 for params per cell

at::Tensor get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, theta);//.reshape({-1,2});
}

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

float integrate_analytical(float x, float t, const float* A, const float xmin, const float xmax, const int nc){
    int cont = 0;

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

        cont++;
        if (cont > nc){
            break;
        }
    }
}

float get_numerical_phi(float x, float t, int nSteps2, const float* A, const float xmin, const float xmax, const int nc){
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

float integrate_numerical(float x, float t, const float* A, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2){
    float xPrev = x;
    float deltaT = t / nSteps1;
    int c = get_cell(x, xmin, xmax, nc);
    for(int j = 0; j < nSteps1; j++) {
        float xTemp = get_psi(xPrev, deltaT, A, xmin, xmax, nc);
        int cTemp = get_cell(xTemp, xmin, xmax, nc);
        float xNum = get_numerical_phi(xPrev, deltaT, nSteps2, A, xmin, xmax, nc);
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

float integrate_analytical_derivative(float x, float t, const float* A, const float xmin, const float xmax, const int nc, std::vector<float> &xr, std::vector<float> &tr){
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
void integrate_analytical_derivative2(float* result, float x, float t, const float* A, const float xmin, const float xmax, const int nc){
    int cont = 0;

    int c;
    float left, right, psi, thit;
    while (true) {
        c = get_cell(x, xmin, xmax, nc);
        left = left_boundary(c, xmin, xmax, nc);
        right = right_boundary(c, xmin, xmax, nc);
        psi = get_psi(x, t, A, xmin, xmax, nc);

        if ((left <= psi) && (psi <= right)){
            result[0] = psi;
            result[1] = t;
            result[2] = c;
            return;
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

        cont++;
        if (cont > nc){
            break;
        }
    }
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

float derivative_total(std::vector<float> &xr, std::vector<float> &tr, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    
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


int sign(int r){
    if (r > 0) return 1;
    if (r < 0) return -1;
    return 0;
}

float derivative_total2(float xini, float tm, int cm, int k, const float* B, const float* A, const float xmin, const float xmax, const int nc){
    
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

at::Tensor integrate_1(at::Tensor points, at::Tensor theta, at::Tensor Bt, float xmin, float xmax, int nc){
    float t = 1.0;

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();
    for(int i = 0; i < n_batch; i++) { // for all batches

        // Precompute affine velocity field
        at::Tensor At = get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            newpoints[i*n_points + j] = integrate_analytical(x[j], t, A, xmin, xmax, nc);
        }
    }
    return output;
}

at::Tensor integrate_2(at::Tensor points, at::Tensor theta, at::Tensor Bt, float xmin, float xmax, int nc, int nSteps1=10, int nSteps2=10){
    float t = 1.0;

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();
    for(int i = 0; i < n_batch; i++) { // for all batches
        
        // Precompute affine velocity field
        at::Tensor At = get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            newpoints[i*n_points + j] = integrate_numerical(x[j], t, A, xmin, xmax, nc, nSteps1, nSteps2);
        }
    }
    return output;
}

at::Tensor derivative_1(at::Tensor points, at::Tensor theta, at::Tensor Bt, float xmin, float xmax, int nc){
    float t = 1.0;

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);
    auto gradpoints = gradient.data_ptr<float>();

    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();
    for(int i = 0; i < n_batch; i++) { // for all batches

        // Precompute affine velocity field
        at::Tensor At = get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();



        for(int j = 0; j < n_points; j++) { // for all points
            std::vector<float> xr, tr;
            newpoints[i*n_points + j] = integrate_analytical_derivative(x[j], t, A, xmin, xmax, nc, xr, tr);
            // TODO: think how are we going to pass xr and tr from forward to backward functions in pytorch
            for(int k = 0; k < d; k++){
                gradpoints[i*(n_points+d) + j*d + k] = derivative_total(xr, tr, k, B, A, xmin, xmax, nc);
            }
        }
    }
    return gradient;
    // return output;
}

at::Tensor derivative_2(at::Tensor points, at::Tensor theta, at::Tensor Bt, float xmin, float xmax, int nc){
    float t = 1.0;

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);
    auto gradpoints = gradient.data_ptr<float>();

    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();
    for(int i = 0; i < n_batch; i++) { // for all batches

        // Precompute affine velocity field
        at::Tensor At = get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            float result[e];
            integrate_analytical_derivative2(result, x[j], t, A, xmin, xmax, nc);
            for(int p = 0; p < e; p++){
                newpoints[i*(n_points + e) + j*e + p] = result[p];
            }
            for(int k = 0; k < d; k++){
                float phi = result[0];
                float t = result[1];
                int c = result[2];
                gradpoints[i*(n_points+d) + j*d + k] = derivative_total2(x[j], t, c, k, B, A, xmin, xmax, nc);
            }
        }
    }
    return gradient;
    // return output;
}



float test(){
    float x = 0.4;
    float t = 1.0;
    float xmin = 0;
    float xmax = 1;
    int nc = 3;
    at::Tensor B = at::ones({2*nc, nc-1});
    torch::Tensor theta = torch::ones({nc-1, 1});
    at::Tensor Am = get_affine(B, theta);
    float* A = Am.data_ptr<float>();
    float v = get_velocity(x, A, xmin, xmax, nc);
    float psi = get_psi(x, t, A, xmin, xmax, nc);
    float thit = get_hit_time(x, A, xmin, xmax, nc);
    float phi = integrate_analytical(x, t, A, xmin, xmax, nc);

    int c = get_cell(x, xmin, xmax, nc);
    float xr = right_boundary(c, xmin, xmax, nc);
    float xl = left_boundary(c, xmin, xmax, nc);
    return phi;
}


void cpab_forward(){

}

void cpab_backward(){
    
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cpab_forward, "Cpab transformer forward");
    m.def("backward", &cpab_backward, "Cpab transformer backward");
    m.def("integrate_1", &integrate_1, "Integrate analytic");
    m.def("integrate_2", &integrate_2, "Integrate numeric");
    m.def("derivative_1", &derivative_1, "Test method");
    m.def("derivative_2", &derivative_2, "Test method");
}




