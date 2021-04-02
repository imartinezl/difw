#include <torch/extension.h>
#include "../../core/cpab.h"
#include <iostream>

// FUNCTIONS

at::Tensor torch_get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, theta);//.reshape({-1,2});
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

    for(int i = 0; i < n_batch; i++) { // for all batches
        
        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            newpoints[i*n_points + j] = get_velocity(x[j], A, xmin, xmax, nc);
        }
    }
    return output;
}


// INTEGRATION
at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
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
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            newpoints[i*n_points + j] = integrate_numeric(x[j], t, A, xmin, xmax, nc, nSteps1, nSteps2);
        }
    }
    return output;
}

at::Tensor torch_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
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
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            newpoints[i*n_points + j] = integrate_closed_form(x[j], t, A, xmin, xmax, nc);
        }
    }
    return output;
}


// DERIVATIVE

// TODO: remove method (two integrations for gradient computation)
at::Tensor torch_derivative_numeric_old(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    float t = 1.0;

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);




    // Allocate output
    const int e = 3;
    // auto output = torch::zeros({n_batch, n_points}, at::kCPU);
    // auto newpoints = output.data_ptr<float>();
    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);
    auto gradpoints = gradient.data_ptr<float>();


    // Convert to pointers
    float* B = Bt.data_ptr<float>();
    float* x = points.data_ptr<float>();
    for(int k = 0; k < d; k++){
        for(int i = 0; i < n_batch; i++) { // for all batches
            // Precompute affine velocity field
            at::Tensor At1 = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
            float* A1 = At1.data_ptr<float>();

            // TODO: review this, make it faster
            auto theta_acc = theta.accessor<float,2>();
            float current = theta_acc[i][k];
            theta.index_put_({i, k}, current + h);
            at::Tensor At2 = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
            float* A2 = At2.data_ptr<float>();
            theta.index_put_({i, k}, current);

            float phi_1, phi_2;
            for(int j = 0; j < n_points; j++) { // for all points
                phi_1 = integrate_numeric(x[j], t, A1, xmin, xmax, nc, nSteps1, nSteps2);
                phi_2 = integrate_numeric(x[j], t, A2, xmin, xmax, nc, nSteps1, nSteps2);

                gradpoints[i*(n_points * d) + j*d + k] = (phi_2 - phi_1)/h;
            }
        }
    }
    return gradient;
}

// TODO: remove method? maybe not, it is useful for gradient method
at::Tensor torch_derivative_numeric(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);

    at::Tensor phi_1 =  torch_integrate_numeric(points, theta, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
}

// TODO: remove method (has std::vector)
at::Tensor torch_derivative_full(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
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
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            std::vector<float> xr, tr;
            newpoints[i*n_points + j] = integrate_closed_form_trace_full(x[j], t, A, xmin, xmax, nc, xr, tr);
            // TODO: think how are we going to pass xr and tr from forward to backward functions in pytorch
            for(int k = 0; k < d; k++){
                gradpoints[i*(n_points * d) + j*d + k] = derivative_phi_theta_full(xr, tr, k, d, B, A, xmin, xmax, nc);
            }
        }
    }
    return gradient;
    // return output;
}

// TODO: remove method? maybe not, it is useful for gradient method
at::Tensor torch_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
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
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            float result[e];
            integrate_closed_form_trace(result, x[j], t, A, xmin, xmax, nc);
            for(int p = 0; p < e; p++){
                newpoints[i*(n_points * e) + j*e + p] = result[p];
            }
            for(int k = 0; k < d; k++){
                float phi = result[0];
                float tm = result[1];
                int cm = result[2];
                gradpoints[i*(n_points * d) + j*d + k] = derivative_phi_theta(x[j], tm, cm, k, d, B, A, xmin, xmax, nc);
            }
        }
    }
    return gradient;
}




// TRANSFORMATION
at::Tensor torch_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc){
    float t = 1.0;

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
    for(int i = 0; i < n_batch; i++) { // for all batches

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
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
    float t = 1.0;

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
    for(int i = 0; i < n_batch; i++) { // for all batches

        // Precompute affine velocity field
        at::Tensor At = torch_get_affine(Bt, theta.index({i, torch::indexing::Slice()}));
        float* A = At.data_ptr<float>();

        for(int j = 0; j < n_points; j++) { // for all points
            for(int k = 0; k < d; k++){ // for all parameters theta
                float phi = newpoints[i*(n_points * e) + j*e + 0];
                float tm = newpoints[i*(n_points * e) + j*e + 1];
                int cm = newpoints[i*(n_points * e) + j*e + 2];
                gradpoints[i*(n_points * d) + j*d + k] = derivative_phi_theta(x[j], tm, cm, k, d, B, A, xmin, xmax, nc);
            }
        }
    }
    return gradient;
    // return output;
}

at::Tensor torch_derivative_numeric_trace(at::Tensor phi_1, at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10, const float h=1e-3){
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);
    const int d = theta.size(1);

    auto gradient = torch::zeros({n_batch, n_points, d}, at::kCPU);

    // at::Tensor phi_1 =  torch_integrate_numeric(points, theta, Bt, xmin, xmax, nc, nSteps1, nSteps2);
    // at::Tensor phi_1 =  torch_integrate_closed_form(points, theta, Bt, xmin, xmax, nc);
    
    for(int k = 0; k < d; k++){
        at::Tensor theta_2 = theta.clone();
        at::Tensor row = theta_2.index({torch::indexing::Slice(), k});
        theta_2.index_put_({torch::indexing::Slice(), k}, row + h);
        at::Tensor phi_2 =  torch_integrate_numeric(points, theta_2, Bt, xmin, xmax, nc, nSteps1, nSteps2);
        // at::Tensor phi_2 =  torch_integrate_closed_form(points, theta_2, Bt, xmin, xmax, nc);
        gradient.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), k}, (phi_2 - phi_1)/h);
    }
    return gradient;
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
}




