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
        // cls token
        for (int d = 0; d < embed_dim; d++) {
            output[idx * embed_dim + d] = weights[cls_off + d];
        }
    } else {
        int patch_idx = t - 1;
        // linear projection
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
    // add position embedding (including cls position)
    for (int d = 0; d < embed_dim; d++) {
        output[idx * embed_dim + d] += weights[pos_off + t * embed_dim + d];
    }
}

__kernel void vit_layernorm(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
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
    for (int d = 0; d < embed_dim; d++) {
        out[d] = (in[d] - mean) * inv_std * weights[gamma_off + d] + biases[beta_off + d];
    }
}

__kernel void vit_attention(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
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
    // Compute Q, K, V for this token
    float q[128], k[128], v[128]; // assume max head_dim <= 128
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

    // Compute attention for this token (just for its own query against all keys)
    // We need to compute softmax over all tokens for each head.
    // For simplicity, we'll compute attention weights for this token and output.
    // But we need to store attention output for this token.
    // We'll compute per head.
    float attn_out[128] = {0};
    for (int h = 0; h < num_heads; h++) {
        float attn_weights[128]; // max total_tokens
        float max_val = -1e10f;
        // compute scores with all keys
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
        // weight sum of values
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int t2 = 0; t2 < total_tokens; t2++) {
                __global const float* value_in = input + (b * total_tokens + t2) * embed_dim;
                val += attn_weights[t2] * value_in[h * head_dim + d];
            }
            attn_out[h * head_dim + d] = val;
        }
    }

    // Project output
    float out_vals[128] = {0};
    for (int d = 0; d < embed_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += attn_out[j] * weights[proj_off + j * embed_dim + d];
        }
        out_vals[d] = sum + biases[proj_b_off + d];
    }
    // Write to output
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

    float hidden[256]; // max mlp_dim
    for (int i = 0; i < mlp_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += in[j] * weights[ffn1_off + j * mlp_dim + i];
        }
        float val = sum + biases[ffn1_b_off + i];
        // GELU activation (approximation)
        float x = val;
        float gelu = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
        hidden[i] = gelu;
    }
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            sum += hidden[j] * weights[ffn2_off + j * embed_dim + i];
        }
        out[i] = sum + biases[ffn2_b_off + i];
    }
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

__kernel void vit_head(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
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

    // Extract CLS token (position 0)
    __global const float* cls = input + b * total_tokens * embed_dim;

    // LayerNorm
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

    // Linear to 1
    float sum = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        sum += normed[d] * weights[head_weight_off + d];
    }
    output[b] = sum + biases[head_bias_off];
}