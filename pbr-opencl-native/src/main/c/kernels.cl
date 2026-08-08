__kernel void forward_kernel(
    __global const float* g_input,
    __global const float* g_weights,
    __global const float* g_biases,
    __global float* g_output,
    int g_batchSize,
    int g_numLayers,
    __global const int* g_layerDims,
    __global float* g_hidden,
    int g_maxMidSize
) {
    int sampleId = get_global_id(0);
    if (sampleId >= g_batchSize) return;

    int weightOff = 0;
    int biasOff = 0;
    int sampleOffset = sampleId * g_maxMidSize;
    int cumOffset = 0;

    __global const float* in = g_input + sampleId * g_layerDims[0];
    __global float* out = g_hidden + sampleOffset;

    for (int l = 0; l < g_numLayers; l++) {
        int inDim = g_layerDims[l];
        int outDim = g_layerDims[l + 1];
        __global const float* w = g_weights + weightOff;
        __global const float* b = g_biases + biasOff;

        for (int i = 0; i < outDim; i++) {
            float sum = b[i];
            int base = i * inDim;
            for (int j = 0; j < inDim; j++) {
                sum += w[base + j] * in[j];
            }
            out[i] = (l == g_numLayers - 1) ? sum : (sum > 0 ? sum : 0);
        }

        in = out;
        cumOffset += outDim;
        if (l < g_numLayers - 1) {
            out = g_hidden + sampleOffset + cumOffset;
        }
        weightOff += inDim * outDim;
        biasOff += outDim;
    }

    int outDim = g_layerDims[g_numLayers];
    for (int i = 0; i < outDim; i++) {
        g_output[sampleId * outDim + i] = in[i];
    }
}

__kernel void backward_kernel(
    __global const float* g_input,
    __global const float* g_weights,
    __global const float* g_label,
    __global const float* g_gradOutput,
    __global float* g_gradWeights,
    __global float* g_gradBiases,
    int g_batchSize,
    int g_numLayers,
    __global const int* g_layerDims,
    __global float* g_hidden,
    int g_maxMidSize
) {
    int sampleId = get_global_id(0);
    if (sampleId >= g_batchSize) return;

    int outDim = g_layerDims[g_numLayers];
    int sampleOffset = sampleId * g_maxMidSize;

    int totalWeightOff = 0;
    int totalBiasOff = 0;
    for (int l = 0; l < g_numLayers; l++) {
        totalWeightOff += g_layerDims[l] * g_layerDims[l + 1];
        totalBiasOff += g_layerDims[l + 1];
    }

    int cumOffset = 0;
    for (int l = 0; l < g_numLayers; l++) {
        cumOffset += g_layerDims[l + 1];
    }

    __global float* currGrad = (__global float*)(g_gradOutput + sampleId * outDim);

    for (int l = g_numLayers - 1; l >= 0; l--) {
        int inDim = g_layerDims[l];
        int outDimL = g_layerDims[l + 1];

        cumOffset -= outDimL;

        int wOff = totalWeightOff - g_layerDims[l] * g_layerDims[l + 1];
        int bOff = totalBiasOff - g_layerDims[l + 1];

        __global const float* w = g_weights + wOff;
        __global float* gw = g_gradWeights + wOff;
        __global float* gb = g_gradBiases + bOff;

        __global const float* prevAct = (l == 0) ? (g_input + sampleId * inDim) : (g_hidden + sampleOffset + cumOffset);

        for (int i = 0; i < outDimL; i++) {
            float g = currGrad[i];
            gb[i] += g;
            for (int j = 0; j < inDim; j++) {
                gw[i * inDim + j] += g * prevAct[j];
            }
        }

        if (l > 0) {
            __global float* prevGrad = g_hidden + sampleOffset + cumOffset;
            for (int j = 0; j < inDim; j++) {
                float sum = 0.0f;
                for (int i = 0; i < outDimL; i++) {
                    sum += w[i * inDim + j] * currGrad[i];
                }
                prevGrad[j] = sum * (prevAct[j] > 0 ? 1.0f : 0.0f);
            }
            currGrad = prevGrad;
        }

        totalWeightOff -= inDim * outDimL;
        totalBiasOff -= outDimL;
    }
}

__kernel void update_kernel(
    __global float* g_weights,
    __global float* g_biases,
    __global const float* g_gradWeights,
    __global const float* g_gradBiases,
    __global float* g_vWeights,
    __global float* g_vBiases,
    int g_batchSize,
    float g_lr,
    float g_momentum,
    int g_totalWeights,
    int g_totalBiases
) {
    int idx = get_global_id(0);
    if (idx < g_totalWeights) {
        float avg = g_gradWeights[idx] / (float)g_batchSize;
        float v = g_momentum * g_vWeights[idx] - g_lr * avg;
        g_vWeights[idx] = v;
        g_weights[idx] += v;
    } else if (idx < g_totalWeights + g_totalBiases) {
        int bIdx = idx - g_totalWeights;
        float avg = g_gradBiases[bIdx] / (float)g_batchSize;
        float v = g_momentum * g_vBiases[bIdx] - g_lr * avg;
        g_vBiases[bIdx] = v;
        g_biases[bIdx] += v;
    }
}