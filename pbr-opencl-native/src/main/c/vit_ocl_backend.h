#ifndef VIT_OCL_BACKEND_H
#define VIT_OCL_BACKEND_H

#include <CL/cl.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel kernel_embed;
    cl_kernel kernel_layernorm;
    cl_kernel kernel_attention;
    cl_kernel kernel_ffn;
    cl_kernel kernel_head;
    cl_kernel kernel_add;

    cl_mem d_input;
    cl_mem d_output;
    cl_mem d_weights;
    cl_mem d_biases;
    cl_mem d_token_embeds;
    cl_mem d_layernorm_input;
    cl_mem d_attn_output;
    cl_mem d_ffn_output;
    cl_mem d_head_input;

    float* host_weights;
    float* host_biases;

    int embed_dim;
    int num_layers;
    int num_heads;
    int mlp_dim;
    int seq_len;
    int in_channels;
    int total_weights;
    int total_biases;
    int max_batch;
    int initialized;
} VitBackend;

VitBackend* vit_backend_create(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, long seed);
VitBackend* vit_backend_create_with_weights(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, const float* weights, const float* biases);
void vit_backend_destroy(VitBackend* backend);

void vit_backend_forward(VitBackend* backend, const float* input, float* output, int batch_size);

void vit_backend_get_weights(VitBackend* backend, float* out);
void vit_backend_get_biases(VitBackend* backend, float* out);
void vit_backend_set_weights(VitBackend* backend, const float* weights);
void vit_backend_set_biases(VitBackend* backend, const float* biases);
int vit_backend_get_total_weights(VitBackend* backend);
int vit_backend_get_total_biases(VitBackend* backend);

#endif