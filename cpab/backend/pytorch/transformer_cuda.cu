#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../core/cpab_ops.cuh"

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

at::Tensor cuda_get_cell(at::Tensor points, const float xmin, const float xmax, const int nc, at::Tensor output){

   // Problem size
   const int n_points = points.size(0);
   
   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0));
   dim3 tpb(256);

   // Launch kernel
   kernel_get_cell<<<bc, tpb>>>(n_points, points.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<int>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_get_velocity(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_get_velocity<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_derivative_velocity_dtheta(at::Tensor points, at::Tensor theta, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int d = Bt.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), d);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_get_velocity<<<bc, tpb>>>(n_points, d, 
      points.data_ptr<float>(), Bt.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_derivative_velocity_dx(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_velocity_dx<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}


at::Tensor cuda_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_numeric<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), t, xmin, xmax, nc, nSteps1, nSteps2, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_closed_form<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), t, xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}



at::Tensor cuda_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_closed_form<<<bc, tpb>>>(n_points, n_batch, d,
      points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), t, xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}

at::Tensor cuda_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_closed_form_trace<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), t, xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}


at::Tensor cuda_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_closed_form_trace<<<bc, tpb>>>(n_points, n_batch, d,
      output.data_ptr<float>(), points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}


at::Tensor cuda_interpolate_grid_forward(at::Tensor points, at::Tensor output){

   // Problem size
   const int n_batch = points.size(0);
   const int n_points = points.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_interpolate_grid_forward<<<bc, tpb>>>(n_points, n_batch, points.data_ptr<float>(), output.data_ptr<float>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_interpolate_grid_backward(at::Tensor grad_prev, at::Tensor points, at::Tensor output){

   // Problem size
   const int n_batch = points.size(0);
   const int n_points = points.size(1);
   
   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_interpolate_grid_backward<<<bc, tpb>>>(n_points, n_batch, grad_prev.data_ptr<float>(), points.data_ptr<float>(), output.data_ptr<float>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

// GRADIENT SPACE

at::Tensor cuda_derivative_space_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_space_closed_form<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), t, xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}

at::Tensor cuda_derivative_space_closed_form_dtheta(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_space_closed_form_dtheta<<<bc, tpb>>>(n_points, n_batch, d,
      points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), t, xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}



at::Tensor cuda_derivative_space_closed_form_dx(at::Tensor points, at::Tensor theta, at::Tensor At, const float t, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(1);
   const int n_batch = theta.size(0);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_space_closed_form_dx<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), t, xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}