#include <torch/extension.h>

// Cuda forward declaration
// at::Tensor cpab_cuda_forward(at::Tensor points_in, at::Tensor trels_in,  
//                              at::Tensor nstepsolver_in, at::Tensor nc_in, 
// 							 const int broadcast, at::Tensor output);
// at::Tensor cpab_cuda_backward(at::Tensor points_in, at::Tensor As_in, 
//                               at::Tensor Bs_in, at::Tensor nstepsolver_in,
//                               at::Tensor nc, const int broadcast, at::Tensor output);
at::Tensor cuda_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output);
at::Tensor cuda_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, at::Tensor output);

                              
// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// FUNCTIONS

at::Tensor torch_get_affine(at::Tensor B, at::Tensor theta){
    return at::matmul(B, at::transpose(theta, 0, 1));//.reshape({-1,2});
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
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);
    
    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);
    at::Tensor At = torch_get_affine(Bt, theta);
    
    // Call kernel launcher
    return cuda_get_velocity(points, theta, At, xmin, xmax, nc, output);
}


// INTEGRATION

at::Tensor torch_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, const int nSteps1=10, const int nSteps2=10){
    // Do input checking
    CHECK_INPUT(points);
    CHECK_INPUT(theta);
    CHECK_INPUT(Bt);

    // Problem size
    const int n_points = points.size(0);
    const int n_batch = theta.size(0);

    // Allocate output
    auto output = torch::zeros({n_batch, n_points}, at::kCUDA);
    at::Tensor At = torch_get_affine(Bt, theta);

    // Call kernel launcher
    return cuda_integrate_numeric(points, theta, At, xmin, xmax, nc, nSteps1, nSteps2, output);
}



// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("forward", &cpab_forward, "Cpab transformer forward (CUDA)");
    // m.def("backward", &cpab_backward, "Cpab transformer backward (CUDA)");
    m.def("get_cell", &torch_get_cell, "Test (CUDA)");
    m.def("get_velocity", &torch_get_velocity, "Test (CUDA)");
    m.def("integrate_numeric", &torch_integrate_numeric, "Test (CUDA)");
}


// Function declaration
// at::Tensor cpab_forward(at::Tensor points_in, //[ndim, n_points] or [batch_size, ndim, n_points]
//                         at::Tensor trels_in,  //[batch_size, nC, ndim, ndim+1]
//                         at::Tensor nstepsolver_in, // scalar
//                         at::Tensor nc_in){ // ndim length tensor
//     // Do input checking
//     CHECK_INPUT(points_in);
//     CHECK_INPUT(trels_in);
//     CHECK_INPUT(nstepsolver_in);
//     CHECK_INPUT(nc_in);
    
//     // Determine if grid is matrix or tensor
//     const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == trels_in.size(0));
    
//     // Problem size
//     const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
//     const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
//     const auto batch_size = trels_in.size(0);
       
//     // Allocate output
//     auto output = torch::zeros({batch_size, ndim, nP}, at::kCUDA); // [batch_size, ndim, nP]   

//     // Call kernel launcher
//     return cpab_cuda_forward(points_in, trels_in, nstepsolver_in, nc_in, 
//                             broadcast, output);
// }

// at::Tensor cpab_backward(at::Tensor points_in, // [ndim, nP] or [batch_size, ndim, n_points]
//                          at::Tensor As_in, // [n_theta, nC, ndim, ndim+1]
//                          at::Tensor Bs_in, // [d, nC, ndim, ndim+1]
//                          at::Tensor nstepsolver_in, // scalar
//                          at::Tensor nc_in){ // ndim length tensor
//     // Do input checking
//     CHECK_INPUT(points_in);
//     CHECK_INPUT(As_in);
//     CHECK_INPUT(Bs_in);
//     CHECK_INPUT(nstepsolver_in);
//     CHECK_INPUT(nc_in);

//     // Determine if grid is matrix or tensor
//     const int broadcast = (int)(points_in.dim() == 3 & points_in.size(0) == As_in.size(0));
    
//     // Problem size
//     const int ndim = (broadcast) ? points_in.size(1) : points_in.size(0);
//     const int nP = (broadcast) ? points_in.size(2) : points_in.size(1);
//     const auto n_theta = As_in.size(0);
//     const auto d = Bs_in.size(0);
    
//     // Allocate output
//     auto output = torch::zeros({d, n_theta, ndim, nP}, at::kCUDA);

//     // Call kernel launcher
//     return cpab_cuda_backward(points_in, As_in, Bs_in, nstepsolver_in, nc_in, 
//                             broadcast, output);
// }


