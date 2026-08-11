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

    cl_kernel kernel_embed_bwd;
    cl_kernel kernel_layernorm_bwd;
    cl_kernel kernel_attention_bwd;
    cl_kernel kernel_ffn_bwd;
    cl_kernel kernel_head_bwd;

    cl_mem d_input;
    cl_mem d_output;
    cl_mem d_weights;
    cl_mem d_biases;
    cl_mem d_gradWeights;
    cl_mem d_gradBiases;
    cl_mem d_vWeights;
    cl_mem d_vBiases;

    cl_mem d_token_embeds;
    cl_mem d_layernorm_input;
    cl_mem d_attn_output;
    cl_mem d_ffn_output;
    cl_mem d_head_input;

    cl_mem d_layernorm_mean;
    cl_mem d_layernorm_var;
    cl_mem d_attn_weights;
    cl_mem d_ffn_hidden;

    float* host_weights;
    float* host_biases;
    float* host_vWeights;
    float* host_vBiases;

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
void vit_backend_backward(VitBackend* backend, const float* input, const float* label, const float* grad_output, int batch_size);
void vit_backend_update(VitBackend* backend, const float* grad_weights, const float* grad_biases, int batch_size, float lr, float momentum);
void vit_backend_zero_gradients(VitBackend* backend);

void vit_backend_get_weights(VitBackend* backend, float* out);
void vit_backend_get_biases(VitBackend* backend, float* out);
void vit_backend_set_weights(VitBackend* backend, const float* weights);
void vit_backend_set_biases(VitBackend* backend, const float* biases);
int vit_backend_get_total_weights(VitBackend* backend);
int vit_backend_get_total_biases(VitBackend* backend);

#endif