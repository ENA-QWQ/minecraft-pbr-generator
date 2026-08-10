#include "vit_ocl_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_BATCH 128
#define MAX_SOURCE_SIZE 65536

static const char* read_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fp = fopen("./pbr-opencl-native/src/main/c/vit_kernels.cl", "r");
    }
    if (!fp) {
        fp = fopen("../pbr-opencl-native/src/main/c/vit_kernels.cl", "r");
    }
    if (!fp) {
        fp = fopen("vit_kernels.cl", "r");
    }
    if (!fp) {
        fprintf(stderr, "Failed to open vit_kernel file\n");
        return NULL;
    }
    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    size_t size = fread(source, 1, MAX_SOURCE_SIZE - 1, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

static int init_opencl(VitBackend* backend) {
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

static int build_program(VitBackend* backend) {
    cl_int err;
    const char* source = read_kernel_source("vit_kernels.cl");
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
        fprintf(stderr, "OpenCL build error (vit): %s\n", log);
        free(log);
        return 0;
    }
    backend->kernel_embed = clCreateKernel(backend->program, "vit_embed", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_layernorm = clCreateKernel(backend->program, "vit_layernorm", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_attention = clCreateKernel(backend->program, "vit_attention", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_ffn = clCreateKernel(backend->program, "vit_ffn", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_head = clCreateKernel(backend->program, "vit_head", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_add = clCreateKernel(backend->program, "vit_add", &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static int allocate_buffers(VitBackend* backend) {
    cl_int err;
    backend->max_batch = MAX_BATCH;

    int seq_len = backend->seq_len;
    int embed_dim = backend->embed_dim;
    int total_tokens = seq_len + 1;

    backend->d_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                      MAX_BATCH * seq_len * backend->in_channels * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_output = clCreateBuffer(backend->context, CL_MEM_WRITE_ONLY,
                                       MAX_BATCH * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                       backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;

    backend->d_token_embeds = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                             MAX_BATCH * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layernorm_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                                MAX_BATCH * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_attn_output = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                            MAX_BATCH * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ffn_output = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                           MAX_BATCH * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_head_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
                                           MAX_BATCH * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static void compute_offsets(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels,
                            int* offsets, int* bias_offsets) {
    int off = 0, b_off = 0;
    // embedding weight: in_channels * embed_dim
    offsets[0] = off; off += in_channels * embed_dim;
    // cls token
    offsets[1] = off; off += embed_dim;
    // pos embed
    offsets[2] = off; off += (seq_len + 1) * embed_dim;
    // bias offsets start after weights
    // no biases for embedding, cls, pos
    bias_offsets[0] = 0; // unused

    for (int l = 0; l < num_layers; l++) {
        // LayerNorm1 gamma
        offsets[3 + l*8 + 0] = off; off += embed_dim;
        // LayerNorm1 beta (bias)
        bias_offsets[1 + l*8 + 0] = b_off; b_off += embed_dim;
        // QKV weight
        offsets[3 + l*8 + 1] = off; off += embed_dim * (3 * embed_dim);
        // QKV bias
        bias_offsets[1 + l*8 + 1] = b_off; b_off += 3 * embed_dim;
        // output proj weight
        offsets[3 + l*8 + 2] = off; off += embed_dim * embed_dim;
        // output proj bias
        bias_offsets[1 + l*8 + 2] = b_off; b_off += embed_dim;
        // LayerNorm2 gamma
        offsets[3 + l*8 + 3] = off; off += embed_dim;
        // LayerNorm2 beta
        bias_offsets[1 + l*8 + 3] = b_off; b_off += embed_dim;
        // FFN1 weight
        offsets[3 + l*8 + 4] = off; off += embed_dim * mlp_dim;
        // FFN1 bias
        bias_offsets[1 + l*8 + 4] = b_off; b_off += mlp_dim;
        // FFN2 weight
        offsets[3 + l*8 + 5] = off; off += mlp_dim * embed_dim;
        // FFN2 bias
        bias_offsets[1 + l*8 + 5] = b_off; b_off += embed_dim;
    }
    // final LN gamma
    offsets[3 + num_layers*8 + 0] = off; off += embed_dim;
    bias_offsets[1 + num_layers*8 + 0] = b_off; b_off += embed_dim;
    // final head linear weight
    offsets[3 + num_layers*8 + 1] = off; off += embed_dim;
    bias_offsets[1 + num_layers*8 + 1] = b_off; b_off += 1;
}

static int init_weights(VitBackend* backend, long seed) {
    backend->host_weights = (float*)malloc(backend->total_weights * sizeof(float));
    if (!backend->host_weights) return 0;
    backend->host_biases = (float*)malloc(backend->total_biases * sizeof(float));
    if (!backend->host_biases) return 0;

    srand((unsigned int)seed);
    for (int i = 0; i < backend->total_weights; i++) {
        backend->host_weights[i] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
    }
    for (int i = 0; i < backend->total_biases; i++) {
        backend->host_biases[i] = 0.0f;
    }
    cl_int err;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
                               backend->total_weights * sizeof(float), backend->host_weights, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
                               backend->total_biases * sizeof(float), backend->host_biases, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

VitBackend* vit_backend_create(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, long seed) {
    VitBackend* backend = (VitBackend*)calloc(1, sizeof(VitBackend));
    if (!backend) return NULL;
    backend->embed_dim = embed_dim;
    backend->num_layers = num_layers;
    backend->num_heads = num_heads;
    backend->mlp_dim = mlp_dim;
    backend->seq_len = seq_len;
    backend->in_channels = in_channels;

    int total_weights = 0, total_biases = 0;
    total_weights += in_channels * embed_dim; // embedding
    total_weights += embed_dim; // cls
    total_weights += (seq_len + 1) * embed_dim; // pos
    for (int l = 0; l < num_layers; l++) {
        total_weights += embed_dim; // ln1 gamma
        total_biases += embed_dim; // ln1 beta
        total_weights += embed_dim * (3 * embed_dim); // qkv
        total_biases += 3 * embed_dim;
        total_weights += embed_dim * embed_dim; // proj
        total_biases += embed_dim;
        total_weights += embed_dim; // ln2 gamma
        total_biases += embed_dim; // ln2 beta
        total_weights += embed_dim * mlp_dim; // ffn1
        total_biases += mlp_dim;
        total_weights += mlp_dim * embed_dim; // ffn2
        total_biases += embed_dim;
    }
    total_weights += embed_dim; // final ln gamma
    total_biases += embed_dim; // final ln beta
    total_weights += embed_dim; // head linear
    total_biases += 1;

    backend->total_weights = total_weights;
    backend->total_biases = total_biases;

    if (!init_opencl(backend)) { vit_backend_destroy(backend); return NULL; }
    if (!build_program(backend)) { vit_backend_destroy(backend); return NULL; }
    if (!allocate_buffers(backend)) { vit_backend_destroy(backend); return NULL; }
    if (!init_weights(backend, seed)) { vit_backend_destroy(backend); return NULL; }
    backend->initialized = 1;
    return backend;
}

VitBackend* vit_backend_create_with_weights(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, const float* weights, const float* biases) {
    VitBackend* backend = vit_backend_create(embed_dim, num_layers, num_heads, mlp_dim, seq_len, in_channels, 0);
    if (!backend) return NULL;
    if (weights) vit_backend_set_weights(backend, weights);
    if (biases) vit_backend_set_biases(backend, biases);
    return backend;
}

void vit_backend_destroy(VitBackend* backend) {
    if (!backend) return;
    if (backend->kernel_embed) clReleaseKernel(backend->kernel_embed);
    if (backend->kernel_layernorm) clReleaseKernel(backend->kernel_layernorm);
    if (backend->kernel_attention) clReleaseKernel(backend->kernel_attention);
    if (backend->kernel_ffn) clReleaseKernel(backend->kernel_ffn);
    if (backend->kernel_head) clReleaseKernel(backend->kernel_head);
    if (backend->kernel_add) clReleaseKernel(backend->kernel_add);
    if (backend->program) clReleaseProgram(backend->program);
    if (backend->queue) clReleaseCommandQueue(backend->queue);
    if (backend->context) clReleaseContext(backend->context);
    if (backend->d_input) clReleaseMemObject(backend->d_input);
    if (backend->d_output) clReleaseMemObject(backend->d_output);
    if (backend->d_weights) clReleaseMemObject(backend->d_weights);
    if (backend->d_biases) clReleaseMemObject(backend->d_biases);
    if (backend->d_token_embeds) clReleaseMemObject(backend->d_token_embeds);
    if (backend->d_layernorm_input) clReleaseMemObject(backend->d_layernorm_input);
    if (backend->d_attn_output) clReleaseMemObject(backend->d_attn_output);
    if (backend->d_ffn_output) clReleaseMemObject(backend->d_ffn_output);
    if (backend->d_head_input) clReleaseMemObject(backend->d_head_input);
    free(backend->host_weights);
    free(backend->host_biases);
    free(backend);
}

void vit_backend_forward(VitBackend* backend, const float* input, float* output, int batch_size) {
    if (!backend || !backend->initialized) return;
    if (batch_size > backend->max_batch) batch_size = backend->max_batch;
    cl_int err;

    int seq_len = backend->seq_len;
    int embed_dim = backend->embed_dim;
    int total_tokens = seq_len + 1;
    int in_channels = backend->in_channels;
    int num_layers = backend->num_layers;
    int num_heads = backend->num_heads;
    int mlp_dim = backend->mlp_dim;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_input, CL_FALSE, 0,
                               batch_size * seq_len * in_channels * sizeof(float), input, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: write input failed\n"); return; }

    int offsets[100], bias_offsets[100];
    compute_offsets(embed_dim, num_layers, num_heads, mlp_dim, seq_len, in_channels, offsets, bias_offsets);

    // Embed
    clSetKernelArg(backend->kernel_embed, 0, sizeof(cl_mem), &backend->d_input);
    clSetKernelArg(backend->kernel_embed, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_embed, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_embed, 3, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_embed, 4, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_embed, 5, sizeof(int), &seq_len);
    clSetKernelArg(backend->kernel_embed, 6, sizeof(int), &in_channels);
    clSetKernelArg(backend->kernel_embed, 7, sizeof(int), &embed_dim);
    clSetKernelArg(backend->kernel_embed, 8, sizeof(int), &total_tokens);
    clSetKernelArg(backend->kernel_embed, 9, sizeof(int), &offsets[0]); // embed weight offset
    clSetKernelArg(backend->kernel_embed, 10, sizeof(int), &offsets[1]); // cls offset
    clSetKernelArg(backend->kernel_embed, 11, sizeof(int), &offsets[2]); // pos offset
    size_t global_embed = batch_size * total_tokens;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_embed, 1, NULL, &global_embed, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: embed kernel failed\n"); return; }

    cl_mem current = backend->d_token_embeds;
    for (int l = 0; l < num_layers; l++) {
        int ln1_gamma_off = offsets[3 + l*8 + 0];
        int ln1_beta_off = bias_offsets[1 + l*8 + 0];
        int qkv_off = offsets[3 + l*8 + 1];
        int qkv_b_off = bias_offsets[1 + l*8 + 1];
        int proj_off = offsets[3 + l*8 + 2];
        int proj_b_off = bias_offsets[1 + l*8 + 2];
        int ln2_gamma_off = offsets[3 + l*8 + 3];
        int ln2_beta_off = bias_offsets[1 + l*8 + 3];
        int ffn1_off = offsets[3 + l*8 + 4];
        int ffn1_b_off = bias_offsets[1 + l*8 + 4];
        int ffn2_off = offsets[3 + l*8 + 5];
        int ffn2_b_off = bias_offsets[1 + l*8 + 5];

        // LN1
        clSetKernelArg(backend->kernel_layernorm, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_layernorm, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm, 3, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm, 4, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm, 5, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm, 6, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm, 7, sizeof(int), &ln1_gamma_off);
        clSetKernelArg(backend->kernel_layernorm, 8, sizeof(int), &ln1_beta_off);
        size_t global_ln = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm, 1, NULL, &global_ln, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: layernorm1 failed\n"); return; }

        // Attention
        clSetKernelArg(backend->kernel_attention, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_attention, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_attention, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_attention, 3, sizeof(cl_mem), &backend->d_attn_output);
        clSetKernelArg(backend->kernel_attention, 4, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_attention, 5, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_attention, 6, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_attention, 7, sizeof(int), &num_heads);
        clSetKernelArg(backend->kernel_attention, 8, sizeof(int), &qkv_off);
        clSetKernelArg(backend->kernel_attention, 9, sizeof(int), &qkv_b_off);
        clSetKernelArg(backend->kernel_attention, 10, sizeof(int), &proj_off);
        clSetKernelArg(backend->kernel_attention, 11, sizeof(int), &proj_b_off);
        size_t global_attn = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention, 1, NULL, &global_attn, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: attention kernel failed\n"); return; }

        // Add residual: current = current + attn_output
        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_attn_output);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current); // output same as first
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        size_t global_add = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: add residual failed\n"); return; }

        // LN2
        clSetKernelArg(backend->kernel_layernorm, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_layernorm, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm, 3, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm, 4, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm, 5, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm, 6, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm, 7, sizeof(int), &ln2_gamma_off);
        clSetKernelArg(backend->kernel_layernorm, 8, sizeof(int), &ln2_beta_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm, 1, NULL, &global_ln, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: layernorm2 failed\n"); return; }

        // FFN
        clSetKernelArg(backend->kernel_ffn, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_ffn, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_ffn, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_ffn, 3, sizeof(cl_mem), &backend->d_ffn_output);
        clSetKernelArg(backend->kernel_ffn, 4, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_ffn, 5, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_ffn, 6, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_ffn, 7, sizeof(int), &mlp_dim);
        clSetKernelArg(backend->kernel_ffn, 8, sizeof(int), &ffn1_off);
        clSetKernelArg(backend->kernel_ffn, 9, sizeof(int), &ffn1_b_off);
        clSetKernelArg(backend->kernel_ffn, 10, sizeof(int), &ffn2_off);
        clSetKernelArg(backend->kernel_ffn, 11, sizeof(int), &ffn2_b_off);
        size_t global_ffn = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_ffn, 1, NULL, &global_ffn, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: ffn kernel failed\n"); return; }

        // Add residual again
        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_ffn_output);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: add residual2 failed\n"); return; }
    }

    // Head: extract CLS token (index 0) from current
    int final_ln_gamma = offsets[3 + num_layers*8 + 0];
    int final_ln_beta = bias_offsets[1 + num_layers*8 + 0];
    int head_weight = offsets[3 + num_layers*8 + 1];
    int head_bias = bias_offsets[1 + num_layers*8 + 1];
    clSetKernelArg(backend->kernel_head, 0, sizeof(cl_mem), &current);
    clSetKernelArg(backend->kernel_head, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_head, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_head, 3, sizeof(cl_mem), &backend->d_output);
    clSetKernelArg(backend->kernel_head, 4, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_head, 5, sizeof(int), &total_tokens);
    clSetKernelArg(backend->kernel_head, 6, sizeof(int), &embed_dim);
    clSetKernelArg(backend->kernel_head, 7, sizeof(int), &final_ln_gamma);
    clSetKernelArg(backend->kernel_head, 8, sizeof(int), &final_ln_beta);
    clSetKernelArg(backend->kernel_head, 9, sizeof(int), &head_weight);
    clSetKernelArg(backend->kernel_head, 10, sizeof(int), &head_bias);
    size_t global_head = batch_size;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_head, 1, NULL, &global_head, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: head kernel failed\n"); return; }

    err = clEnqueueReadBuffer(backend->queue, backend->d_output, CL_TRUE, 0,
                              batch_size * sizeof(float), output, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "vit forward: read output failed\n"); return; }
}

void vit_backend_get_weights(VitBackend* backend, float* out) {
    if (!backend || !out) return;
    memcpy(out, backend->host_weights, backend->total_weights * sizeof(float));
}

void vit_backend_get_biases(VitBackend* backend, float* out) {
    if (!backend || !out) return;
    memcpy(out, backend->host_biases, backend->total_biases * sizeof(float));
}

void vit_backend_set_weights(VitBackend* backend, const float* weights) {
    if (!backend || !weights) return;
    memcpy(backend->host_weights, weights, backend->total_weights * sizeof(float));
    clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
                         backend->total_weights * sizeof(float), backend->host_weights, 0, NULL, NULL);
}

void vit_backend_set_biases(VitBackend* backend, const float* biases) {
    if (!backend || !biases) return;
    memcpy(backend->host_biases, biases, backend->total_biases * sizeof(float));
    clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
                         backend->total_biases * sizeof(float), backend->host_biases, 0, NULL, NULL);
}

int vit_backend_get_total_weights(VitBackend* backend) {
    return backend ? backend->total_weights : 0;
}

int vit_backend_get_total_biases(VitBackend* backend) {
    return backend ? backend->total_biases : 0;
}