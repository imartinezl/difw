#ifndef CPAB_OPS_GPU
#define CPAB_OPS_GPU


__device__ int get_cell(float x, const float xmin, const float xmax, const int nc);
__device__ float get_velocity(float x, const float* A, const int n_batch, const int batch_index, const float xmin, const float xmax, const int nc);


__global__ void kernel_get_cell(const int n_points, const float* points, 
    const float xmin, const float xmax, const int nc, int* newpoints);
__global__ void kernel_get_velocity(const int n_points, const int n_batch, 
    const float *points, const float *A, 
    const float xmin, const float xmax, const int nc, float* newpoints);

__global__ void kernel_integrate_numeric(const int n_points, const int n_batch, 
    const float *points, const float *A, 
    const float xmin, const float xmax, const int nc, const int nSteps1, const int nSteps2, 
    float* newpoints);


// __device__ int cuda_mymin(int a, double b);
// __device__ double cuda_fmod(double numer, double denom);
// __device__ int cuda_findcellidx_1D(const float* p, const int nx);
// __device__ int cuda_findcellidx_2D(const float* p, const int nx, const int ny);
// __device__ int cuda_findcellidx_3D(const float* p, const int nx, const int ny, const int nz);
// __device__ void A_times_b_1D(float x[], const float* A, float* b);
// __device__ void A_times_b_2D(float x[], const float* A, float* b);
// __device__ void A_times_b_3D(float x[], const float* A, float* b);
// __device__ void A_times_b_linear_1D(float x[], const float* A, float* b);
// __device__ void A_times_b_linear_2D(float x[], const float* A, float* b);
// __device__ void A_times_b_linear_3D(float x[], const float* A, float* b);
// __global__ void cpab_cuda_kernel_forward_1D(const int nP, const int batch_size,
//                                             float* newpoints, const float* points,
//                                             const float* Trels, const int* nStepSolver,
//                                             const int* nc, const int broadcast);
// __global__ void cpab_cuda_kernel_forward_2D(const int nP, const int batch_size,
//                                             float* newpoints, const float* points,
//                                             const float* Trels, const int* nStepSolver,
//                                             const int* nc, const int broadcast);
// __global__ void cpab_cuda_kernel_forward_3D(const int nP, const int batch_size,
//                                             float* newpoints, const float* points, 
//                                             const float* Trels, const int* nStepSolver,
//                                             const int* nc, const int broadcast);
// __global__ void cpab_cuda_kernel_backward_1D(dim3 nthreads, const int n_theta, 
//                                              const int d, const int nP, const int nC,
//                                              float* grad, const float* points, 
//                                              const float* As, const float* Bs,
//                                              const int* nStepSolver, const int* nc, 
//                                              const int broadcast);
// __global__ void cpab_cuda_kernel_backward_2D(dim3 nthreads, const int n_theta, 
//                                              const int d, const int nP, const int nC,
//                                              float* grad, const float* points, 
//                                              const float* As, const float* Bs,
//                                              const int* nStepSolver, const int* nc, 
//                                              const int broadcast);
// __global__ void cpab_cuda_kernel_backward_3D(dim3 nthreads, const int n_theta, 
//                                              const int d, const int nP, const int nC,
//                                              float* grad, const float* points, 
//                                              const float* As, const float* Bs,
//                                              const int* nStepSolver, const int* nc, 
//                                              const int broadcast);


#endif
