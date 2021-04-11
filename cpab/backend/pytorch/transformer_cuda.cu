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
   const int n_points = points.size(0);
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

at::Tensor cuda_integrate_numeric(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, at::Tensor output){
   // Problem size
   const int n_points = points.size(0);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_numeric<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), xmin, xmax, nc, nSteps1, nSteps2, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}

at::Tensor cuda_integrate_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(0);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_closed_form<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}



at::Tensor cuda_derivative_closed_form(at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(0);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch); //, d);
   dim3 tpb(256, 1); //, 1);

   // Launch kernel
   kernel_derivative_closed_form<<<bc, tpb>>>(n_points, n_batch, d,
      points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}

at::Tensor cuda_integrate_closed_form_trace(at::Tensor points, at::Tensor theta, at::Tensor At, const float xmin, const float xmax, const int nc, at::Tensor output){
   // Problem size
   const int n_points = points.size(0);
   const int n_batch = theta.size(0);

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_integrate_closed_form_trace<<<bc, tpb>>>(n_points, n_batch, 
      points.data_ptr<float>(), At.data_ptr<float>(), xmin, xmax, nc, output.data_ptr<float>());

   
   gpuErrchk( cudaPeekAtLastError() );                           
   return output; 
}


at::Tensor cuda_derivative_closed_form_trace(at::Tensor output, at::Tensor points, at::Tensor theta, at::Tensor At, at::Tensor Bt, const float xmin, const float xmax, const int nc, at::Tensor gradient){
   // Problem size
   const int n_points = points.size(0);
   const int n_batch = theta.size(0);
   const int d = theta.size(1);

   // Kernel configuration
   // dim3 bc((int)ceil(n_points/256.0), n_batch, d);
   // dim3 tpb(256, 1, 1);

   // Launch kernel
   // kernel_derivative_closed_form_trace<<<bc, tpb>>>(n_points, n_batch, d,
   //    output.data_ptr<float>(), points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), xmin, xmax, nc, gradient.data_ptr<float>());

   // Kernel configuration
   dim3 bc((int)ceil(n_points/256.0), n_batch);
   dim3 tpb(256, 1);

   // Launch kernel
   kernel_derivative_closed_form_trace_optimized<<<bc, tpb>>>(n_points, n_batch, d,
      output.data_ptr<float>(), points.data_ptr<float>(), At.data_ptr<float>(), Bt.data_ptr<float>(), xmin, xmax, nc, gradient.data_ptr<double>());

   gpuErrchk( cudaPeekAtLastError() );                           
   return gradient; 
}