__kernel void vit_embed(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    int batch_size,
    int seq_len,
    int in_channels,
    int embed_dim,
    int total_tokens,
    int embed_weight_off,
    int cls_off,
    int pos_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;

    if (t == 0) {
        for (int d = 0; d < embed_dim; d++) {
            output[idx * embed_dim + d] = weights[cls_off + d];
        }
    } else {
        int patch_idx = t - 1;
        __global const float* in_ptr = input + b * seq_len * in_channels + patch_idx * in_channels;
        __global float* out_ptr = output + idx * embed_dim;
        for (int d = 0; d < embed_dim; d++) {
            float sum = 0.0f;
            for (int c = 0; c < in_channels; c++) {
                sum += in_ptr[c] * weights[embed_weight_off + c * embed_dim + d];
            }
            out_ptr[d] = sum;
        }
    }
    for (int d = 0; d < embed_dim; d++) {
        output[idx * embed_dim + d] += weights[pos_off + t * embed_dim + d];
    }
}

__kernel void vit_layernorm(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* mean_out,
    __global float* var_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int gamma_off,
    int beta_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global float* out = output + idx * embed_dim;
    float mean = 0.0f, var = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        mean += in[d];
    }
    mean /= embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float diff = in[d] - mean;
        var += diff * diff;
    }
    var /= embed_dim;
    float inv_std = 1.0f / sqrt(var + 1e-6f);
    mean_out[idx] = mean;
    var_out[idx] = inv_std;
    for (int d = 0; d < embed_dim; d++) {
        out[d] = (in[d] - mean) * inv_std * weights[gamma_off + d] + biases[beta_off + d];
    }
}

__kernel void vit_attention(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* attn_weights_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_heads,
    int qkv_off,
    int qkv_b_off,
    int proj_off,
    int proj_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int head_dim = embed_dim / num_heads;
    float q[128], k[128], v[128];
    __global const float* in = input + idx * embed_dim;
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int q_idx = h * head_dim + d;
            int k_idx = embed_dim + h * head_dim + d;
            int v_idx = 2 * embed_dim + h * head_dim + d;
            float qsum = 0.0f, ksum = 0.0f, vsum = 0.0f;
            for (int j = 0; j < embed_dim; j++) {
                qsum += in[j] * weights[qkv_off + j * (3 * embed_dim) + q_idx];
                ksum += in[j] * weights[qkv_off + j * (3 * embed_dim) + k_idx];
                vsum += in[j] * weights[qkv_off + j * (3 * embed_dim) + v_idx];
            }
            q[h * head_dim + d] = qsum + biases[qkv_b_off + q_idx];
            k[h * head_dim + d] = ksum + biases[qkv_b_off + k_idx];
            v[h * head_dim + d] = vsum + biases[qkv_b_off + v_idx];
        }
    }
    float attn_out[128] = {0};
    for (int h = 0; h < num_heads; h++) {
        float attn_weights[128];
        float max_val = -1e10f;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            __global const float* key_in = input + (b * total_tokens + t2) * embed_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[h * head_dim + d] * key_in[h * head_dim + d];
            }
            attn_weights[t2] = score / sqrt((float)head_dim);
            if (attn_weights[t2] > max_val) max_val = attn_weights[t2];
        }
        float sum_exp = 0.0f;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            attn_weights[t2] = exp(attn_weights[t2] - max_val);
            sum_exp += attn_weights[t2];
        }
        for (int t2 = 0; t2 < total_tokens; t2++) {
            attn_weights[t2] /= sum_exp;
        }
        for (int t2 = 0; t2 < total_tokens; t2++) {
            __global float* w_ptr = attn_weights_out + (b * total_tokens * total_tokens) + t * total_tokens + t2;
            w_ptr[h] = attn_weights[t2];
        }
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int t2 = 0; t2 < total_tokens; t2++) {
                __global const float* value_in = input + (b * total_tokens + t2) * embed_dim;
                val += attn_weights[t2] * value_in[h * head_dim + d];
            }
            attn_out[h * head_dim + d] = val;
        }
    }
    float out_vals[128];
    for (int d = 0; d < embed_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += attn_out[j] * weights[proj_off + j * embed_dim + d];
        }
        out_vals[d] = sum + biases[proj_b_off + d];
    }
    __global float* out = output + idx * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        out[d] = out_vals[d];
    }
}

__kernel void vit_ffn(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* hidden_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int mlp_dim,
    int ffn1_off,
    int ffn1_b_off,
    int ffn2_off,
    int ffn2_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global float* out = output + idx * embed_dim;
    float hidden[256];
    for (int i = 0; i < mlp_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += in[j] * weights[ffn1_off + j * mlp_dim + i];
        }
        float val = sum + biases[ffn1_b_off + i];
        float x = val;
        float gelu = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
        hidden[i] = gelu;
    }
    for (int i = 0; i < mlp_dim; i++) {
        hidden_out[idx * mlp_dim + i] = hidden[i];
    }
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            sum += hidden[j] * weights[ffn2_off + j * embed_dim + i];
        }
        out[i] = sum + biases[ffn2_b_off + i];
    }
}

__kernel void vit_head(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* cls_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int ln_gamma_off,
    int ln_beta_off,
    int head_weight_off,
    int head_bias_off
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;
    __global const float* cls = input + b * total_tokens * embed_dim;
    float mean = 0.0f, var = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        mean += cls[d];
    }
    mean /= embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float diff = cls[d] - mean;
        var += diff * diff;
    }
    var /= embed_dim;
    float inv_std = 1.0f / sqrt(var + 1e-6f);
    float normed[128];
    for (int d = 0; d < embed_dim; d++) {
        normed[d] = (cls[d] - mean) * inv_std * weights[ln_gamma_off + d] + biases[ln_beta_off + d];
    }
    for (int d = 0; d < embed_dim; d++) {
        cls_out[b * embed_dim + d] = normed[d];
    }
    float sum = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        sum += normed[d] * weights[head_weight_off + d];
    }
    output[b] = sum + biases[head_bias_off];
}

__kernel void vit_add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    int batch_size,
    int total_tokens,
    int embed_dim
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens * embed_dim;
    if (idx >= total) return;
    out[idx] = a[idx] + b[idx];
}

__kernel void vit_embed_bwd(
    __global const float* grad_in,
    __global const float* weights,
    __global float* grad_weights,
    __global const float* input,
    int batch_size,
    int seq_len,
    int in_channels,
    int embed_dim,
    int total_tokens,
    int embed_weight_off,
    int cls_off,
    int pos_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int emb_off = idx * embed_dim;
    if (t == 0) {
        for (int d = 0; d < embed_dim; d++) {
            grad_weights[cls_off + d] += grad_in[emb_off + d];
        }
        return;
    }
    int patch_idx = t - 1;
    __global const float* in_ptr = input + b * seq_len * in_channels + patch_idx * in_channels;
    for (int d = 0; d < embed_dim; d++) {
        float g = grad_in[emb_off + d];
        for (int c = 0; c < in_channels; c++) {
            grad_weights[embed_weight_off + c * embed_dim + d] += g * in_ptr[c];
        }
    }
    for (int d = 0; d < embed_dim; d++) {
        grad_weights[pos_off + t * embed_dim + d] += grad_in[emb_off + d];
    }
}

__kernel void vit_layernorm_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_in,
    __global const float* input,
    __global const float* mean,
    __global const float* inv_std,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int gamma_off,
    int beta_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    __global const float* in = input + idx * embed_dim;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;
    float m = mean[idx];
    float istd = inv_std[idx];
    float sum_g = 0.0f, sum_gx = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        sum_g += go[d];
        sum_gx += go[d] * (in[d] - m);
    }
    float norm_factor = istd / embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float x_hat = (in[d] - m) * istd;
        gi[d] = (go[d] * istd) - norm_factor * (sum_g + x_hat * sum_gx);
        grad_weights[gamma_off + d] += go[d] * x_hat;
        grad_biases[beta_off + d] += go[d];
    }
}

__kernel void vit_attention_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global const float* attn_weights,
    __global float* grad_in,
    __global const float* input,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_heads,
    int ln_gamma_off,
    int ln_beta_off,
    int qkv_off,
    int qkv_b_off,
    int proj_off,
    int proj_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int head_dim = embed_dim / num_heads;
    __global const float* in = input + idx * embed_dim;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;

    float proj_grad[128];
    for (int d = 0; d < embed_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += go[j] * weights[proj_off + d * embed_dim + j];
        }
        proj_grad[d] = sum;
    }
    for (int d = 0; d < embed_dim; d++) {
        for (int j = 0; j < embed_dim; j++) {
            grad_weights[proj_off + d * embed_dim + j] += go[d] * input[idx * embed_dim + j];
        }
        grad_biases[proj_b_off + d] += go[d];
    }

    for (int h = 0; h < num_heads; h++) {
        float dv[128];
        for (int d = 0; d < head_dim; d++) {
            dv[d] = 0.0f;
        }
        for (int t2 = 0; t2 < total_tokens; t2++) {
            float weight = attn_weights[(b * total_tokens + t) * total_tokens + t2 * num_heads + h];
            for (int d = 0; d < head_dim; d++) {
                dv[d] += weight * proj_grad[h * head_dim + d];
            }
        }
        for (int t2 = 0; t2 < total_tokens; t2++) {
            float w = attn_weights[(b * total_tokens + t) * total_tokens + t2 * num_heads + h];
            float dq = 0.0f, dk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dq += proj_grad[h * head_dim + d] * in[t2 * embed_dim + h * head_dim + d];
                dk += proj_grad[h * head_dim + d] * in[t * embed_dim + h * head_dim + d];
            }
            dq *= (1.0f / sqrt(head_dim));
            dk *= (1.0f / sqrt(head_dim));
            for (int d = 0; d < head_dim; d++) {
                grad_in[idx * embed_dim + h * head_dim + d] += dq;
            }
            for (int d = 0; d < head_dim; d++) {
                grad_in[(b * total_tokens + t2) * embed_dim + h * head_dim + d] += dk;
            }
        }
        for (int d = 0; d < head_dim; d++) {
            for (int j = 0; j < embed_dim; j++) {
                int q_idx = h * head_dim + d;
                int k_idx = embed_dim + h * head_dim + d;
                int v_idx = 2 * embed_dim + h * head_dim + d;
                grad_weights[qkv_off + j * (3 * embed_dim) + q_idx] += proj_grad[h * head_dim + d] * in[j];
                grad_weights[qkv_off + j * (3 * embed_dim) + k_idx] += 0.0f;
                grad_weights[qkv_off + j * (3 * embed_dim) + v_idx] += dv[d] * in[j];
            }
            grad_biases[qkv_b_off + h * head_dim + d] += proj_grad[h * head_dim + d];
            grad_biases[qkv_b_off + embed_dim + h * head_dim + d] += 0.0f;
            grad_biases[qkv_b_off + 2 * embed_dim + h * head_dim + d] += dv[d];
        }
    }
}

__kernel void vit_ffn_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_in,
    __global const float* input,
    __global const float* hidden,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int mlp_dim,
    int ffn1_off,
    int ffn1_b_off,
    int ffn2_off,
    int ffn2_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;
    __global const float* in = input + idx * embed_dim;
    __global const float* hid = hidden + idx * mlp_dim;

    float d_hidden[256];
    for (int j = 0; j < mlp_dim; j++) {
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; d++) {
            sum += go[d] * weights[ffn2_off + j * embed_dim + d];
        }
        d_hidden[j] = sum;
    }
    for (int j = 0; j < mlp_dim; j++) {
        float x = hid[j];
        float gelu_deriv = 0.5f * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x))) +
                           0.5f * x * (1.0f - tanh(0.79788456f * (x + 0.044715f * x * x * x)) * tanh(0.79788456f * (x + 0.044715f * x * x * x))) *
                           (0.79788456f + 0.107032f * x * x);
        d_hidden[j] *= gelu_deriv;
    }
    for (int d = 0; d < embed_dim; d++) {
        for (int j = 0; j < mlp_dim; j++) {
            grad_weights[ffn2_off + j * embed_dim + d] += go[d] * hid[j];
        }
        grad_biases[ffn2_b_off + d] += go[d];
    }
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            sum += d_hidden[j] * weights[ffn1_off + i * mlp_dim + j];
            grad_weights[ffn1_off + i * mlp_dim + j] += d_hidden[j] * in[i];
        }
        grad_biases[ffn1_b_off + i] += sum;
    }
    for (int d = 0; d < embed_dim; d++) {
        gi[d] = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            gi[d] += d_hidden[j] * weights[ffn1_off + d * mlp_dim + j];
        }
    }
}

__kernel void vit_head_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_cls,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int ln_gamma_off,
    int ln_beta_off,
    int head_weight_off,
    int head_bias_off
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;
    float g = grad_out[b];
    grad_biases[head_bias_off] += g;
    for (int d = 0; d < embed_dim; d++) {
        grad_weights[head_weight_off + d] += g * grad_cls[b * embed_dim + d];
    }
    for (int d = 0; d < embed_dim; d++) {
        grad_cls[b * embed_dim + d] = g * weights[head_weight_off + d];
    }
}