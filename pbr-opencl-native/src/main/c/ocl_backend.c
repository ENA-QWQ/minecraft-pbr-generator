#include "ocl_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT __attribute__((visibility("default")))
#endif

#define MAX_SOURCE_SIZE 65536
#define MAX_BATCH 65536

static const char* read_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fp = fopen("./pbr-opencl-native/src/main/c/kernels.cl", "r");
    }
    if (!fp) {
        fp = fopen("../pbr-opencl-native/src/main/c/kernels.cl", "r");
    }
    if (!fp) {
        fp = fopen("kernels.cl", "r");
    }
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file\n");
        return NULL;
    }
    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    size_t size = fread(source, 1, MAX_SOURCE_SIZE - 1, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

static int init_opencl(OCLBackend* backend) {
    cl_int err;
    err = clGetPlatformIDs(1, &backend->platform, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clGetDeviceIDs(backend->platform, CL_DEVICE_TYPE_GPU, 1, &backend->device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(backend->platform, CL_DEVICE_TYPE_CPU, 1, &backend->device, NULL);
        if (err != CL_SUCCESS) return 0;
    }
    backend->context = clCreateContext(NULL, 1, &backend->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->queue = clCreateCommandQueue(backend->context, backend->device, 0, &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static int build_program(OCLBackend* backend) {
    cl_int err;
    const char* source = read_kernel_source("kernels.cl");
    if (!source) return 0;
    backend->program = clCreateProgramWithSource(backend->context, 1, &source, NULL, &err);
    free((void*)source);
    if (err != CL_SUCCESS) return 0;
    err = clBuildProgram(backend->program, 1, &backend->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL build error: %s\n", log);
        free(log);
        return 0;
    }
    backend->kernel_forward = clCreateKernel(backend->program, "forward_kernel", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_backward = clCreateKernel(backend->program, "backward_kernel", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_update = clCreateKernel(backend->program, "update_kernel", &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static int allocate_buffers(OCLBackend* backend) {
    cl_int err;
    backend->max_batch = MAX_BATCH;

    backend->d_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                       backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_gradWeights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                            backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_gradBiases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                           backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_vWeights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                         backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_vBiases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                        backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_input = clCreateBuffer(backend->context, CL_MEM_READ_ONLY,
                                      MAX_BATCH * backend->feature_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_label = clCreateBuffer(backend->context, CL_MEM_READ_ONLY,
                                      MAX_BATCH * backend->label_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_output = clCreateBuffer(backend->context, CL_MEM_WRITE_ONLY,
                                       MAX_BATCH * backend->label_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layerDims = clCreateBuffer(backend->context, CL_MEM_READ_ONLY,
                                          (backend->num_layers + 1) * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) return 0;

    backend->max_mid_size = 0;
    for (int i = 1; i < backend->num_layers; i++) {
        backend->max_mid_size += backend->layer_sizes[i];
    }

    backend->d_hidden = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                       MAX_BATCH * backend->max_mid_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;

    backend->host_weights = (float*)malloc(backend->total_weights * sizeof(float));
    if (!backend->host_weights) return 0;
    backend->host_biases = (float*)malloc(backend->total_biases * sizeof(float));
    if (!backend->host_biases) return 0;
    backend->host_vWeights = (float*)calloc(backend->total_weights, sizeof(float));
    if (!backend->host_vWeights) return 0;
    backend->host_vBiases = (float*)calloc(backend->total_biases, sizeof(float));
    if (!backend->host_vBiases) return 0;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_vWeights, CL_TRUE, 0,
                               backend->total_weights * sizeof(float), backend->host_vWeights, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_vBiases, CL_TRUE, 0,
                               backend->total_biases * sizeof(float), backend->host_vBiases, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;

    return 1;
}

static int init_weights(OCLBackend* backend, long seed) {
    cl_int err;
    srand((unsigned int)seed);
    int offset = 0;
    for (int l = 0; l < backend->num_layers; l++) {
        int inDim = backend->layer_sizes[l];
        int outDim = backend->layer_sizes[l + 1];
        float std = sqrtf(2.0f / inDim);
        for (int i = 0; i < outDim; i++) {
            for (int j = 0; j < inDim; j++) {
                float u1 = (float)rand() / RAND_MAX;
                float u2 = (float)rand() / RAND_MAX;
                float g = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
                backend->host_weights[offset++] = g * std;
            }
        }
        for (int i = 0; i < outDim; i++) {
            backend->host_biases[offset - outDim + i] = 0.0f;
        }
    }
    err = clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
                               backend->total_weights * sizeof(float), backend->host_weights, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
                               backend->total_biases * sizeof(float), backend->host_biases, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_layerDims, CL_TRUE, 0,
                               (backend->num_layers + 1) * sizeof(int), backend->layer_sizes, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

DLL_EXPORT OCLBackend* ocl_backend_create(const int* layer_sizes, int num_layers, long seed) {
    OCLBackend* backend = (OCLBackend*)calloc(1, sizeof(OCLBackend));
    if (!backend) return NULL;
    backend->num_layers = num_layers;
    backend->layer_sizes = (int*)malloc((num_layers + 1) * sizeof(int));
    if (!backend->layer_sizes) { free(backend); return NULL; }
    memcpy(backend->layer_sizes, layer_sizes, (num_layers + 1) * sizeof(int));
    backend->feature_dim = layer_sizes[0];
    backend->label_dim = layer_sizes[num_layers];
    backend->total_weights = 0;
    backend->total_biases = 0;
    for (int l = 0; l < num_layers; l++) {
        backend->total_weights += layer_sizes[l] * layer_sizes[l + 1];
        backend->total_biases += layer_sizes[l + 1];
    }
    if (!init_opencl(backend)) { ocl_backend_destroy(backend); return NULL; }
    if (!build_program(backend)) { ocl_backend_destroy(backend); return NULL; }
    if (!allocate_buffers(backend)) { ocl_backend_destroy(backend); return NULL; }
    if (!init_weights(backend, seed)) { ocl_backend_destroy(backend); return NULL; }
    backend->initialized = 1;
    return backend;
}

DLL_EXPORT OCLBackend* ocl_backend_create_with_weights(const int* layer_sizes, int num_layers,
                                            const float* weights, const float* biases) {
    OCLBackend* backend = ocl_backend_create(layer_sizes, num_layers, 0);
    if (!backend) return NULL;
    if (weights) ocl_backend_set_weights(backend, weights);
    if (biases) ocl_backend_set_biases(backend, biases);
    return backend;
}

DLL_EXPORT void ocl_backend_destroy(OCLBackend* backend) {
    if (!backend) return;
    if (backend->kernel_forward) clReleaseKernel(backend->kernel_forward);
    if (backend->kernel_backward) clReleaseKernel(backend->kernel_backward);
    if (backend->kernel_update) clReleaseKernel(backend->kernel_update);
    if (backend->program) clReleaseProgram(backend->program);
    if (backend->queue) clReleaseCommandQueue(backend->queue);
    if (backend->context) clReleaseContext(backend->context);
    if (backend->d_weights) clReleaseMemObject(backend->d_weights);
    if (backend->d_biases) clReleaseMemObject(backend->d_biases);
    if (backend->d_input) clReleaseMemObject(backend->d_input);
    if (backend->d_label) clReleaseMemObject(backend->d_label);
    if (backend->d_output) clReleaseMemObject(backend->d_output);
    if (backend->d_gradWeights) clReleaseMemObject(backend->d_gradWeights);
    if (backend->d_gradBiases) clReleaseMemObject(backend->d_gradBiases);
    if (backend->d_vWeights) clReleaseMemObject(backend->d_vWeights);
    if (backend->d_vBiases) clReleaseMemObject(backend->d_vBiases);
    if (backend->d_layerDims) clReleaseMemObject(backend->d_layerDims);
    if (backend->d_hidden) clReleaseMemObject(backend->d_hidden);
    free(backend->host_weights);
    free(backend->host_biases);
    free(backend->host_vWeights);
    free(backend->host_vBiases);
    free(backend->layer_sizes);
    free(backend);
}

DLL_EXPORT void ocl_backend_forward(OCLBackend* backend, const float* input, float* output, int batch_size) {
    if (!backend || !backend->initialized) {
        fprintf(stderr, "ocl_backend_forward: backend not initialized\n");
        return;
    }
    cl_int err;
    if (batch_size > backend->max_batch) batch_size = backend->max_batch;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_input, CL_FALSE, 0,
                               batch_size * backend->feature_dim * sizeof(float), input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "forward: write input failed %d\n", err); return; }

    clSetKernelArg(backend->kernel_forward, 0, sizeof(cl_mem), &backend->d_input);
    clSetKernelArg(backend->kernel_forward, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_forward, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_forward, 3, sizeof(cl_mem), &backend->d_output);
    clSetKernelArg(backend->kernel_forward, 4, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_forward, 5, sizeof(int), &backend->num_layers);
    clSetKernelArg(backend->kernel_forward, 6, sizeof(cl_mem), &backend->d_layerDims);
    clSetKernelArg(backend->kernel_forward, 7, sizeof(cl_mem), &backend->d_hidden);
    clSetKernelArg(backend->kernel_forward, 8, sizeof(int), &backend->max_mid_size);

    size_t global_size = batch_size;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_forward, 1, NULL,
                                 &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "forward: execute kernel failed %d\n", err); return; }

    err = clEnqueueReadBuffer(backend->queue, backend->d_output, CL_TRUE, 0,
                              batch_size * backend->label_dim * sizeof(float), output, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "forward: read output failed %d\n", err); return; }
}

DLL_EXPORT void ocl_backend_backward(OCLBackend* backend, const float* input, const float* label,
                          const float* grad_output, int batch_size) {
    if (!backend || !backend->initialized) {
        fprintf(stderr, "ocl_backend_backward: backend not initialized\n");
        return;
    }
    cl_int err;
    if (batch_size > backend->max_batch) batch_size = backend->max_batch;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_input, CL_FALSE, 0,
                               batch_size * backend->feature_dim * sizeof(float), input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "backward: write input failed %d\n", err); return; }

    err = clEnqueueWriteBuffer(backend->queue, backend->d_label, CL_FALSE, 0,
                               batch_size * backend->label_dim * sizeof(float), label, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "backward: write label failed %d\n", err); return; }

    err = clEnqueueWriteBuffer(backend->queue, backend->d_output, CL_FALSE, 0,
                               batch_size * backend->label_dim * sizeof(float), grad_output, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "backward: write grad_output failed %d\n", err); return; }

    clSetKernelArg(backend->kernel_backward, 0, sizeof(cl_mem), &backend->d_input);
    clSetKernelArg(backend->kernel_backward, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_backward, 2, sizeof(cl_mem), &backend->d_label);
    clSetKernelArg(backend->kernel_backward, 3, sizeof(cl_mem), &backend->d_output);
    clSetKernelArg(backend->kernel_backward, 4, sizeof(cl_mem), &backend->d_gradWeights);
    clSetKernelArg(backend->kernel_backward, 5, sizeof(cl_mem), &backend->d_gradBiases);
    clSetKernelArg(backend->kernel_backward, 6, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_backward, 7, sizeof(int), &backend->num_layers);
    clSetKernelArg(backend->kernel_backward, 8, sizeof(cl_mem), &backend->d_layerDims);
    clSetKernelArg(backend->kernel_backward, 9, sizeof(cl_mem), &backend->d_hidden);
    clSetKernelArg(backend->kernel_backward, 10, sizeof(int), &backend->max_mid_size);

    size_t global_size = batch_size;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_backward, 1, NULL,
                                 &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "backward: execute kernel failed %d\n", err); return; }

    clFinish(backend->queue);
}

DLL_EXPORT void ocl_backend_update(OCLBackend* backend, const float* grad_weights, const float* grad_biases,
                        int batch_size, float lr, float momentum) {
    if (!backend || !backend->initialized) return;
    cl_int err;

    int total = backend->total_weights + backend->total_biases;
    clSetKernelArg(backend->kernel_update, 0, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_update, 1, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_update, 2, sizeof(cl_mem), &backend->d_gradWeights);
    clSetKernelArg(backend->kernel_update, 3, sizeof(cl_mem), &backend->d_gradBiases);
    clSetKernelArg(backend->kernel_update, 4, sizeof(cl_mem), &backend->d_vWeights);
    clSetKernelArg(backend->kernel_update, 5, sizeof(cl_mem), &backend->d_vBiases);
    clSetKernelArg(backend->kernel_update, 6, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_update, 7, sizeof(float), &lr);
    clSetKernelArg(backend->kernel_update, 8, sizeof(float), &momentum);
    clSetKernelArg(backend->kernel_update, 9, sizeof(int), &backend->total_weights);
    clSetKernelArg(backend->kernel_update, 10, sizeof(int), &backend->total_biases);

    size_t global_size = total;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_update, 1, NULL,
                                 &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "update: execute kernel failed %d\n", err); return; }
    clFinish(backend->queue);
    
}

DLL_EXPORT void ocl_backend_zero_gradients(OCLBackend* backend) {
    if (!backend || !backend->initialized) return;
    cl_int err;
    float zero = 0.0f;
    err = clEnqueueFillBuffer(backend->queue, backend->d_gradWeights, &zero, sizeof(float),
                              0, backend->total_weights * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        err = clEnqueueWriteBuffer(backend->queue, backend->d_gradWeights, CL_TRUE, 0,
                                   backend->total_weights * sizeof(float), &zero, 0, NULL, NULL);
    }
    err = clEnqueueFillBuffer(backend->queue, backend->d_gradBiases, &zero, sizeof(float),
                              0, backend->total_biases * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        err = clEnqueueWriteBuffer(backend->queue, backend->d_gradBiases, CL_TRUE, 0,
                                   backend->total_biases * sizeof(float), &zero, 0, NULL, NULL);
    }
    clFinish(backend->queue);
    
}

DLL_EXPORT void ocl_backend_get_weights(OCLBackend* backend, float* out) {
    if (!backend || !backend->initialized || !out) return;
    clEnqueueReadBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
                        backend->total_weights * sizeof(float), out, 0, NULL, NULL);
}

DLL_EXPORT void ocl_backend_get_biases(OCLBackend* backend, float* out) {
    if (!backend || !backend->initialized || !out) return;
    clEnqueueReadBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
                        backend->total_biases * sizeof(float), out, 0, NULL, NULL);
}

DLL_EXPORT void ocl_backend_set_weights(OCLBackend* backend, const float* weights) {
    if (!backend || !backend->initialized || !weights) return;
    clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
                         backend->total_weights * sizeof(float), weights, 0, NULL, NULL);
}

DLL_EXPORT void ocl_backend_set_biases(OCLBackend* backend, const float* biases) {
    if (!backend || !backend->initialized || !biases) return;
    clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
                         backend->total_biases * sizeof(float), biases, 0, NULL, NULL);
}

DLL_EXPORT int ocl_backend_get_total_weights(OCLBackend* backend) {
    return backend ? backend->total_weights : 0;
}

DLL_EXPORT int ocl_backend_get_total_biases(OCLBackend* backend) {
    return backend ? backend->total_biases : 0;
}