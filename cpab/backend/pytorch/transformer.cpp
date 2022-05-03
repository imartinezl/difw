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
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();    
    
    // Problem size
    const int n_points = points.size(1);
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
            newpoints[i*n_points + j] = get_velocity(x[i*n_points + j], A, xmin, xmax, nc);
        }
    }
    return output;
}


at::Tensor torch_derivative_velocity_dtheta(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(1), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto output = torch::zeros({d, n_points}, at::kCPU).contiguous();
    float* newpoints = output.data_ptr<float>();

    // Convert to pointers
    float* x = points.data_ptr<float>();

    // For all theta dimensions
    for(int k = 0; k < d; k++){
        
        // Precompute affine velocity field
        at::Tensor At = Bt.index({torch::indexing::Slice(), k}).contiguous();
        float* A = At.data_ptr<float>();

        // For all points
        for(int j = 0; j < n_points; j++) {
            newpoints[k*n_points + j] = get_velocity(x[k*n_points + j], A, xmin, xmax, nc);
        }
    }
    return output;
}

at::Tensor torch_derivative_velocity_dx(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();

    // Problem size
    const int n_points = points.size(1);
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
            newpoints[i*n_points + j] = derivative_velocity_dx(x[i*n_points + j], A, xmin, xmax, nc);
        }
    }
    return output;
}


// INTEGRATION
at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
    // Batch grid
    if(points.dim() == 1) points = torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
            newpoints[i*n_points + j] = integrate_numeric(x[i*n_points + j], t, A, xmin, xmax, nc, nSteps1, nSteps2);
        }
    }
    return output;
}

at::Tensor torch_integrate_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    if(points.dim() == 1) points = torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
            newpoints[i*n_points + j] = integrate_closed_form(x[i*n_points + j], t, A, xmin, xmax, nc);
        }
    }
    return output;
}


// DERIVATIVE

at::Tensor torch_derivative_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    if(points.dim() == 1) points = torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
            integrate_closed_form_trace(result, x[i*n_points + j], t, A, xmin, xmax, nc);
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];
            // NEW METHOD
            float dphi_dtheta[d];
            derivative_phi_theta(dphi_dtheta, x[i*n_points + j], tm, cm, d, B, A, xmin, xmax, nc);

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
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();

    // Problem size
    const int n_points = points.size(1);
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
            integrate_closed_form_trace(result, x[i*n_points + j], t, A, xmin, xmax, nc);
            for(int p = 0; p < e; p++){
                newpoints[i*(n_points * e) + j*e + p] = result[p];
            }
        }
    }
    return output;
}

at::Tensor torch_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
            derivative_phi_theta(dphi_dtheta, x[i*n_points + j], tm, cm, d, B, A, xmin, xmax, nc);

            // For all parameters theta
            for(int k = 0; k < d; k++){
                gradpoints[i*(n_points * d) + j*d + k] = dphi_dtheta[k];
            }
        }
    }
    return gradient;
}

at::Tensor torch_derivative_numeric_trace(at::Tensor phi_1, at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
at::Tensor torch_interpolate_grid_forward(at::Tensor points){
    const int n_batch = points.size(0);
    const int n_points = points.size(1);

    auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    float* y = output.data_ptr<float>();

    float* x = points.data_ptr<float>();
    
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

at::Tensor torch_interpolate_grid_backward(at::Tensor grad_prev, at::Tensor points){
    const int n_batch = points.size(0);
    const int n_points = points.size(1);

    auto output = torch::zeros({n_batch, n_points}, at::kCPU).contiguous();
    float* gradient = output.data_ptr<float>();

    float* x = points.data_ptr<float>();
    float* g = grad_prev.data_ptr<float>();
    
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

            gradient[n_points*i + x0] += (1-xd) * g[pos];
            gradient[n_points*i + x1] += xd * g[pos];
            gradient[n_points*i + j] += (n_points-1)*(y1-y0) * g[pos];
        }
    }
    return output;
}


// GRADIENT SPACE



at::Tensor torch_derivative_space_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    // auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    // auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points}, at::kCPU);
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
            integrate_closed_form_trace(result, x[i*n_points + j], t, A, xmin, xmax, nc);
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];

            float dphi_dx = derivative_phi_x(x[i*n_points + j], t, tm, cm, A, xmin, xmax, nc);

            gradpoints[i*n_points + j] = dphi_dx;
        }
    }
    return gradient;
}

at::Tensor torch_derivative_space_closed_form_dtheta(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
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
            integrate_closed_form_trace(result, x[i*n_points + j], t, A, xmin, xmax, nc);
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];

            float dphi_dx_dtheta[d];
            derivative_phi_x_theta(dphi_dx_dtheta, x[i*n_points + j], t, tm, cm, d, B, A, xmin, xmax, nc);

            // For all parameters theta
            for(int k = 0; k < d; k++){
                gradpoints[i*(n_points * d) + j*d + k] = dphi_dx_dtheta[k];
            }
        }
    }
    return gradient;
}

at::Tensor torch_derivative_space_closed_form_dx(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    const int e = 3;
    // auto output = torch::zeros({n_batch, n_points, e}, at::kCPU);
    // auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points}, at::kCPU);
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
            integrate_closed_form_trace(result, x[i*n_points + j], t, A, xmin, xmax, nc);
            // float phi = result[0];
            float tm = result[1];
            int cm = result[2];

            float dphi_dx_dx = derivative_phi_x_x(x[i*n_points + j], t, tm, cm, A, xmin, xmax, nc);

            gradpoints[i*n_points + j] = dphi_dx_dx;
        }
    }
    return gradient;
}


at::Tensor torch_derivative_space_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // at::Tensor phi_1 =  torch_integrate_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_2 =  torch_integrate_numeric(points+h, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    at::Tensor phi_2 =  torch_integrate_closed_form(points+h, theta, t, Bt, xmin, xmax, nc);
    at::Tensor gradient = (phi_2 - phi_1) / h;

    return gradient;
}

at::Tensor torch_derivative_space_numeric_dtheta(at::Tensor phi_1, at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({d, n_batch, n_points}, at::kCPU);

    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta_2.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        // at::Tensor phi_2 =  torch_derivative_space_numeric(points, theta_2, t, Bt, xmin, xmax, nc, nSteps1, nSteps2, h);
        at::Tensor phi_2 =  torch_derivative_space_closed_form(points, theta_2, t, Bt, xmin, xmax, nc);
        gradient.index_put_({k, torch::indexing::Slice(), torch::indexing::Slice()}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

at::Tensor torch_derivative_space_numeric_dx(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // at::Tensor dphi_dx_1 =  torch_derivative_space_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2, h);
    // at::Tensor dphi_dx_2 =  torch_derivative_space_numeric(points+h, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2, h);
    at::Tensor dphi_dx_1 =  torch_derivative_space_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    at::Tensor dphi_dx_2 =  torch_derivative_space_closed_form(points+h, theta, t, Bt, xmin, xmax, nc);
    at::Tensor gradient = (dphi_dx_2 - dphi_dx_1) / h;

    return gradient;
}



// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_cell", &torch_get_cell, "Get cell");
    m.def("get_velocity", &torch_get_velocity, "Get Velocity");
    m.def("derivative_velocity_dtheta", &torch_derivative_velocity_dtheta, "Derivative Velocity dtheta");
    m.def("derivative_velocity_dx", &torch_derivative_velocity_dx, "Derivative Velocity dx");
    m.def("integrate_closed_form", &torch_integrate_closed_form, "Integrate closed form");
    m.def("integrate_numeric", &torch_integrate_numeric, "Integrate numeric");
    m.def("derivative_closed_form", &torch_derivative_closed_form, "Derivative closed form");
    m.def("derivative_numeric", &torch_derivative_numeric, "Derivative numeric");
    m.def("integrate_closed_form_trace", &torch_integrate_closed_form_trace, "Integrate closed form trace");
    m.def("derivative_closed_form_trace", &torch_derivative_closed_form_trace, "Derivative closed form trace");
    m.def("derivative_numeric_trace", &torch_derivative_numeric_trace, "Derivative numeric trace");
    m.def("interpolate_grid_forward", &torch_interpolate_grid_forward, "Interpolate grid forward");
    m.def("interpolate_grid_backward", &torch_interpolate_grid_backward, "Interpolate grid backward");
    m.def("derivative_space_numeric", &torch_derivative_space_numeric, "Derivative space numeric");
    m.def("derivative_space_numeric_dtheta", &torch_derivative_space_numeric_dtheta, "Derivative space numeric dtheta");
    m.def("derivative_space_numeric_dx", &torch_derivative_space_numeric_dx, "Derivative space numeric dx");
    m.def("derivative_space_closed_form", &torch_derivative_space_closed_form, "Derivative space closed form");
    m.def("derivative_space_closed_form_dtheta", &torch_derivative_space_closed_form_dtheta, "Derivative space closed form dtheta");
    m.def("derivative_space_closed_form_dx", &torch_derivative_space_closed_form_dx, "Derivative space closed form dx");
}




