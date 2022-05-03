#include <torch/extension.h>

at::Tensor cuda_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_velocity(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_velocity_dx(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_velocity_dtheta(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, at::Tensor output);
at::Tensor cuda_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient);
at::Tensor cuda_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient);
at::Tensor cuda_interpolate_grid_forward(at::Tensor points, at::Tensor output);
at::Tensor cuda_interpolate_grid_backward(at::Tensor grad_prev, at::Tensor points, at::Tensor output);
at::Tensor cuda_derivative_space_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient);
at::Tensor cuda_derivative_space_closed_form_dtheta(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient);
at::Tensor cuda_derivative_space_closed_form_dx(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient);

// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// FUNCTIONS

at::Tensor torch_get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, at::transpose(theta, 0, 1));
}

at::Tensor torch_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc){
    // Do input checking
    CHECK_INPUT(points);
    
    // Problem size
    const int n_points = points.size(0);

    // Allocate output
    auto output = torch::zeros({n_points}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Call kernel launcher
    return cuda_get_cell(points, xmin, xmax, nc, output);
}

at::Tensor torch_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
        
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);
    
    // Call kernel launcher
    return cuda_get_velocity(points, theta, At, xmin, xmax, nc, output);
}

at::Tensor torch_derivative_velocity_dtheta(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(1), points.size(0)}).contiguous() : points.contiguous();
        
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto output = torch::zeros({d, n_points}, at::kCUDA);

    // Call kernel launcher
    return cuda_derivative_velocity_dtheta(points, theta, Bt, xmin, xmax, nc, output);
}

at::Tensor torch_derivative_velocity_dx(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
        
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);
    
    // Call kernel launcher
    return cuda_derivative_velocity_dx(points, theta, At, xmin, xmax, nc, output);
}


// INTEGRATION

at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
        
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_numeric(points, theta, At, t, xmin, xmax, nc, nSteps1, nSteps2, output);
}

at::Tensor torch_integrate_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
        
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_closed_form(points, theta, At, t, xmin, xmax, nc, output);
}

// DERIVATIVE

at::Tensor torch_derivative_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

    at::Tensor phi_1 =  torch_integrate_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 = torch_integrate_numeric(points, theta_2, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 = torch_integrate_closed_form(points, theta_2, t, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

at::Tensor torch_derivative_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points, d}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_closed_form(points, theta, At, Bt, t, xmin, xmax, nc, gradient);
}


// TRANSFORMATION

at::Tensor torch_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int e = 3;

    // Allocate output
    auto output = torch::zeros({n_batch, n_points, e}, at::kCUDA);

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_closed_form_trace(points, theta, At, t, xmin, xmax, nc, output);
}

at::Tensor torch_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(output);
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points, d}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_closed_form_trace(output, points, theta, At, Bt, xmin, xmax, nc, gradient);
}


at::Tensor torch_derivative_numeric_trace(at::Tensor phi_1, at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(phi_1);
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCUDA);

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


at::Tensor torch_interpolate_grid_forward(at::Tensor points){
    // Do input checking
    CHECK_INPUT(points);
    
    // Problem size
    const int n_batch = points.size(0);
    const int n_points = points.size(1);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Call kernel launcher
    return cuda_interpolate_grid_forward(points, output);
}

at::Tensor torch_interpolate_grid_backward(at::Tensor grad_prev, at::Tensor points){
    // Do input checking
    CHECK_INPUT(grad_prev);
    CHECK_INPUT(points);
    
    // Problem size
    const int n_batch = points.size(0);
    const int n_points = points.size(1);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);

    // Call kernel launcher
    return cuda_interpolate_grid_backward(grad_prev, points, output);
}


// GRADIENT SPACE


at::Tensor torch_derivative_space_closed_form(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_space_closed_form(points, theta, At, t, xmin, xmax, nc, gradient);
}

at::Tensor torch_derivative_space_closed_form_dtheta(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points, d}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_space_closed_form_dtheta(points, theta, At, Bt, t, xmin, xmax, nc, gradient);
}

at::Tensor torch_derivative_space_closed_form_dx(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points}, torch::dtype(torch::kDouble).device(torch::kCUDA));

    // Precompute affine velocity field
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_derivative_space_closed_form_dx(points, theta, At, t, xmin, xmax, nc, gradient);
}


at::Tensor torch_derivative_space_numeric(at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points}, at::kCUDA);
    
    // at::Tensor phi_1 =  torch_integrate_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_2 =  torch_integrate_numeric(points+h, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    at::Tensor phi_2 =  torch_integrate_closed_form(points+h, theta, t, Bt, xmin, xmax, nc);
    gradient = (phi_2 - phi_1) / h;

    return gradient;
}

at::Tensor torch_derivative_space_numeric_dtheta(at::Tensor phi_1, at::Tensor points, at::Tensor theta, const float t, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Batch grid
    points = (points.dim() == 1) ? torch::broadcast_to(points, {theta.size(0), points.size(0)}).contiguous() : points.contiguous();
    
    // Do input checking
    CHECK_INPUT(phi_1);
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({d, n_batch, n_points}, at::kCUDA);
   
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
    
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(1);
    const int n_batch = theta.size(0);

    // Allocate output
    auto gradient = torch::zeros({n_batch, n_points}, at::kCUDA);
    
    // at::Tensor dphi_dx_1 =  torch_derivative_space_numeric(points, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2, h);
    // at::Tensor dphi_dx_2 =  torch_derivative_space_numeric(points+h, theta, t, Bt, xmin, xmax, nc, nSteps1, nSteps2, h);
    at::Tensor dphi_dx_1 =  torch_derivative_space_closed_form(points, theta, t, Bt, xmin, xmax, nc);
    at::Tensor dphi_dx_2 =  torch_derivative_space_closed_form(points+h, theta, t, Bt, xmin, xmax, nc);
    gradient = (dphi_dx_2 - dphi_dx_1) / h;

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