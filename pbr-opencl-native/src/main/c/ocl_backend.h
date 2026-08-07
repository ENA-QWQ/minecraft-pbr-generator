#ifndef OCL_BACKEND_H
#define OCL_BACKEND_H

#include <CL/cl.h>
#include <jni.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_forward;
    cl_kernel kernel_backward;
    cl_kernel kernel_update;

    cl_mem d_weights;
    cl_mem d_biases;
    cl_mem d_input;
    cl_mem d_label;
    cl_mem d_output;
    cl_mem d_gradWeights;
    cl_mem d_gradBiases;
    cl_mem d_layerDims;
    cl_mem d_hidden;

    float* host_weights;
    float* host_biases;
    float* host_gradWeights;
    float* host_gradBiases;

    int* layer_sizes;
    int num_layers;
    int total_weights;
    int total_biases;
    int feature_dim;
    int label_dim;
    int max_batch;
    int max_mid_size;

    int initialized;
} OCLBackend;

OCLBackend* ocl_backend_create(const int* layer_sizes, int num_layers, long seed);
OCLBackend* ocl_backend_create_with_weights(const int* layer_sizes, int num_layers,
                                            const float* weights, const float* biases);
void ocl_backend_destroy(OCLBackend* backend);

void ocl_backend_forward(OCLBackend* backend, const float* input, float* output, int batch_size);
void ocl_backend_backward(OCLBackend* backend, const float* input, const float* label,
                          const float* grad_output, int batch_size);
void ocl_backend_update(OCLBackend* backend, const float* grad_weights, const float* grad_biases,
                        int batch_size, float lr, float momentum);
void ocl_backend_zero_gradients(OCLBackend* backend);
void ocl_backend_get_weights(OCLBackend* backend, float* out);
void ocl_backend_get_biases(OCLBackend* backend, float* out);
void ocl_backend_set_weights(OCLBackend* backend, const float* weights);
void ocl_backend_set_biases(OCLBackend* backend, const float* biases);
int ocl_backend_get_total_weights(OCLBackend* backend);
int ocl_backend_get_total_biases(OCLBackend* backend);

#endif