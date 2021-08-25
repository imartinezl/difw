#include <torch/extension.h>
#include "../../core/cpab_ops.h"
#include <iostream>

// FUNCTIONS

at::Tensor torch_get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, theta);
}

at::Tensor torch_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc){
    const int n_points = points.size(0);
    float* x = points.data_ptr<float>();

    auto output = torch::zeros({n_points}, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto newpoints = output.data_ptr<int>();

    for(int i = 0; i < n_points; i++) {
        newpoints[i] = get_cell(x[i], xmin, xmax, nc);
    }
    return output;
}

at::Tensor torch_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {
        
        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            newpoints[i*n_points + j] = get_velocity(x[j], A, xmin, xmax, nc);
        }
    }
    return output;
}


// INTEGRATION
at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {
        
        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            newpoints[i*n_points + j] = integrate_numeric(x[j], t, A, xmin, xmax, nc, nSteps1, nSteps2);
        }
    }
    return output;
}

at::Tensor torch_integrate_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            newpoints[i*n_points + j] = integrate_closed_form(x[j], t, A, xmin, xmax, nc);
        }
    }
    return output;
}


// DERIVATIVE

at::Tensor torch_derivative_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);

    at::Tensor phi_1 =  torch_integrate_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, t, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

at::Tensor torch_derivative_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    // auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    // auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);
    auto gradpoints = gradient.data_ptr<float>();

    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) { 
            float result[e];
            integrate_closed_form_trace(result, x[j], t, A, xmin, xmax, nc);
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];
            // NEW METHOD
            float dphi_dtheta[d];
            derivative_phi_theta(dphi_dtheta, x[j], tm, cm, d, B, A, xmin, xmax, nc);

            // For all parameters theta
            for(int k = 0; k < d; k++){ 
                gradpoints[i*(n_points * d) + j*d + k] = dphi_dtheta[k];
            }
        }
    }
    return gradient;
}

// TRANSFORMATION
at::Tensor torch_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    auto newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            float result[e];
            integrate_closed_form_trace(result, x[j], t, A, xmin, xmax, nc);
            for(int p = 0; p < e; p++){
                newpoints[i*(n_points * e) + j*e + p] = result[p];
            }
        }
    }
    return output;
}

at::Tensor torch_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    // auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);
    auto gradpoints = gradient.data_ptr<float>();

    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();

    // For all batches
    for(int i = 0; i < n_batch; i++) {

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            // float phi = newpoints[i*(n_points * e) + j*e + 0];
            float tm = newpoints[i*(n_points * e) + j*e + 1];
            int cm = newpoints[i*(n_points * e) + j*e + 2];

            float dphi_dtheta[d];
            derivative_phi_theta(dphi_dtheta, x[j], tm, cm, d, B, A, xmin, xmax, nc);

            // For all parameters theta
            for(int k = 0; k < d; k++){
                gradpoints[i*(n_points * d) + j*d + k] = dphi_dtheta[k];
            }
        }
    }
    return gradient;
}

at::Tensor torch_derivative_numeric_trace(at::Tensor phi_1, at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);

    // at::Tensor phi_1 =  torch_integrate_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta_2.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, t, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

float clip(int num, int lower, int upper) {
  return std::max(lower, std::min(num, upper));
}

// Interpolate
at::Tensor torch_interpolate_grid_forward(at::Tensor data){
    const int n_batch = data.size(0);
    const int n_points = data.size(1);

    auto output = torch::zeros({6, n_batch, n_points}, at::kCPU);//.contiguous();
    // auto output = torch::empty_like(data, at::kCPU).contiguous();
    float* y = output.data_ptr<float>();

    float* x = data.data_ptr<float>();
    
    for (int i = 0; i < n_batch; i++)
    {
        for (int j = 0; j < n_points; j++)
        {
            float xc = x[i*n_points + j]*(n_points - 1);
            int x0 = (int) std::floor(xc);
            int x1 = x0 + 1;
            x0 = clip(x0, 0, n_points-1);
            x1 = clip(x1, 0, n_points-1);
            float y0 = x[i*n_points + x0];
            float y1 = x[i*n_points + x1];
            float xd = (float) xc - x0;

            y[0*(n_batch*n_points) + n_points*i + j] = y0 * (1 - xd) + y1 * xd;
            y[1*(n_batch*n_points) + n_points*i + j] = y0;
            y[2*(n_batch*n_points) + n_points*i + j] = y1;
            y[3*(n_batch*n_points) + n_points*i + j] = x0;
            y[4*(n_batch*n_points) + n_points*i + j] = x1;
            y[5*(n_batch*n_points) + n_points*i + j] = xd;
        }
        
    }
    return output;
}

at::Tensor torch_interpolate_grid_backward(at::Tensor g, at::Tensor y, at::Tensor y0, at::Tensor y1, at::Tensor x0, at::Tensor x1, at::Tensor xd){
    const int n_batch = g.size(0);
    const int n_points = g.size(1);

    auto output = torch::zeros({n_batch, n_points}, at::kCPU).contiguous();
    float* grad = output.data_ptr<float>();

    float* y0_arr = y0.data_ptr<float>();
    float* y1_arr = y1.data_ptr<float>();
    float* x0_arr = x0.data_ptr<float>();
    float* x1_arr = x1.data_ptr<float>();
    float* xd_arr = xd.data_ptr<float>();
    float* g_arr = g.data_ptr<float>();
    
    for (int i = 0; i < n_batch; i++)
    {
        for (int j = 0; j < n_points; j++)
        {
            int pos = n_points*i + j;

            int row = x0_arr[pos];
            float value = 1-xd_arr[pos];
            grad[n_points*i + row] += value * g_arr[pos];

            row = x1_arr[pos];
            value = xd_arr[pos];
            grad[n_points*i + row] += value * g_arr[pos];

            row = j;
            value = (n_points-1)*(y1_arr[pos]-y0_arr[pos]);
            grad[n_points*i + row] += value * g_arr[pos];
        }
    }
    return output;
}

// Interpolate
at::Tensor torch_interpolate_grid_forward_new(at::Tensor data){
    const int n_batch = data.size(0);
    const int n_points = data.size(1);

    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    float* y = output.data_ptr<float>();

    float* x = data.data_ptr<float>();
    
    for (int i = 0; i < n_batch; i++)
    {
        for (int j = 0; j < n_points; j++)
        {
            float xc = x[i*n_points + j]*(n_points - 1);
            int x0 = (int) std::floor(xc);
            int x1 = x0 + 1;
            x0 = clip(x0, 0, n_points-1);
            x1 = clip(x1, 0, n_points-1);
            float y0 = x[i*n_points + x0];
            float y1 = x[i*n_points + x1];
            float xd = (float) xc - x0;

            y[n_points*i + j] = y0 * (1 - xd) + y1 * xd;
        }
        
    }
    return output;
}

at::Tensor torch_interpolate_grid_backward_new(at::Tensor g, at::Tensor data){
    const int n_batch = g.size(0);
    const int n_points = g.size(1);

    auto output = torch::zeros({n_batch, n_points}, at::kCPU).contiguous();
    float* grad = output.data_ptr<float>();

    float* x = data.data_ptr<float>();
    float* g_arr = g.data_ptr<float>();
    
    for (int i = 0; i < n_batch; i++)
    {
        for (int j = 0; j < n_points; j++)
        {
            int pos = n_points*i + j;
            
            float xc = x[pos]*(n_points - 1);
            int x0 = (int) std::floor(xc);
            int x1 = x0 + 1;
            x0 = clip(x0, 0, n_points-1);
            x1 = clip(x1, 0, n_points-1);
            float y0 = x[i*n_points + x0];
            float y1 = x[i*n_points + x1];
            float xd = (float) xc - x0;

            grad[n_points*i + x0] += (1-xd) * g_arr[pos];
            grad[n_points*i + x1] += xd * g_arr[pos];
            grad[n_points*i + j] += (n_points-1)*(y1-y0) * g_arr[pos];
        }
    }
    return output;
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_cell", &torch_get_cell, "Get cell");
    m.def("get_velocity", &torch_get_velocity, "Get Velocity");
    m.def("integrate_closed_form", &torch_integrate_closed_form, "Integrate closed form");
    m.def("integrate_numeric", &torch_integrate_numeric, "Integrate numeric");
    m.def("derivative_closed_form", &torch_derivative_closed_form, "Derivative closed form");
    m.def("derivative_numeric", &torch_derivative_numeric, "Derivative numeric");
    m.def("integrate_closed_form_trace", &torch_integrate_closed_form_trace, "Integrate closed form trace");
    m.def("derivative_closed_form_trace", &torch_derivative_closed_form_trace, "Derivative closed form trace");
    m.def("derivative_numeric_trace", &torch_derivative_numeric_trace, "Derivative numeric trace");
    m.def("interpolate_grid_forward", &torch_interpolate_grid_forward, "Interpolate grid forward");
    m.def("interpolate_grid_backward", &torch_interpolate_grid_backward, "Interpolate grid backward");
    m.def("interpolate_grid_forward_new", &torch_interpolate_grid_forward_new, "Interpolate grid forward");
    m.def("interpolate_grid_backward_new", &torch_interpolate_grid_backward_new, "Interpolate grid backward");
}




