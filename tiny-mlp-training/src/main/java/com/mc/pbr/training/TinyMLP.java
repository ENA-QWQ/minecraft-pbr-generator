package com.mc.pbr.training;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class TinyMLP implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int[] layerSizes;
    private final float[][][] weights;
    private final float[][] biases;
    private final float[][][] vWeights;
    private final float[][] vBiases;

    public TinyMLP(int[] layerSizes, Random rng) {
        this.layerSizes = layerSizes;
        int numLayers = layerSizes.length - 1;
        weights = new float[numLayers][][];
        biases = new float[numLayers][];
        vWeights = new float[numLayers][][];
        vBiases = new float[numLayers][];

        for (int l = 0; l < numLayers; l++) {
            int inDim = layerSizes[l];
            int outDim = layerSizes[l + 1];
            weights[l] = new float[outDim][inDim];
            biases[l] = new float[outDim];
            vWeights[l] = new float[outDim][inDim];
            vBiases[l] = new float[outDim];

            float std = (float) Math.sqrt(2.0 / inDim);
            for (int i = 0; i < outDim; i++) {
                for (int j = 0; j < inDim; j++) {
                    weights[l][i][j] = (float) rng.nextGaussian() * std;
                }
                biases[l][i] = 0.0f;
            }
        }
    }

    public int[] getLayerSizes() {
        return layerSizes;
    }

    public void forward(float[] input, float[][] activations, float[][] zs) {
        int numLayers = layerSizes.length - 1;
        activations[0] = input;
        for (int l = 0; l < numLayers; l++) {
            int inDim = layerSizes[l];
            int outDim = layerSizes[l + 1];
            float[] aPrev = activations[l];
            float[] z = zs[l];
            float[] a = activations[l + 1];

            for (int i = 0; i < outDim; i++) {
                float sum = biases[l][i];
                float[] wRow = weights[l][i];
                for (int j = 0; j < inDim; j++) {
                    sum += wRow[j] * aPrev[j];
                }
                z[i] = sum;
                if (l == numLayers - 1) {
                    a[i] = sum;
                } else {
                    a[i] = sum > 0 ? sum : 0;
                }
            }
        }
    }

    public void backward(float[] target, float[][] activations, float[][] zs,
                         float[][] deltas,
                         float[][][] gradWeights, float[][] gradBiases) {
        int numLayers = layerSizes.length - 1;
        int lastIdx = numLayers - 1;
        float[] outAct = activations[lastIdx + 1];
        float[] deltaOut = deltas[lastIdx];
        for (int i = 0; i < deltaOut.length; i++) {
            deltaOut[i] = outAct[i] - target[i];
        }

        for (int l = lastIdx - 1; l >= 0; l--) {
            float[] z = zs[l];
            float[] deltaNext = deltas[l + 1];
            float[] delta = deltas[l];
            float[][] nextW = weights[l + 1];
            int nextOutDim = deltaNext.length;
            for (int j = 0; j < delta.length; j++) {
                float sum = 0;
                for (int k = 0; k < nextOutDim; k++) {
                    sum += nextW[k][j] * deltaNext[k];
                }
                delta[j] = z[j] > 0 ? sum : 0;
            }
        }

        for (int l = 0; l < numLayers; l++) {
            float[] aPrev = activations[l];
            float[] delta = deltas[l];
            float[][] gradW = gradWeights[l];
            float[] gradB = gradBiases[l];
            for (int i = 0; i < delta.length; i++) {
                float d = delta[i];
                gradB[i] += d;
                float[] gRow = gradW[i];
                for (int j = 0; j < aPrev.length; j++) {
                    gRow[j] += d * aPrev[j];
                }
            }
        }
    }

    public void update(float[][][] gradWeights, float[][] gradBiases,
                       int batchSize, float lr, float momentum) {
        float invBatch = 1.0f / batchSize;
        for (int l = 0; l < weights.length; l++) {
            float[][] w = weights[l];
            float[][] gw = gradWeights[l];
            float[][] vw = vWeights[l];
            float[] b = biases[l];
            float[] gb = gradBiases[l];
            float[] vb = vBiases[l];

            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[i].length; j++) {
                    float avgGrad = gw[i][j] * invBatch;
                    vw[i][j] = momentum * vw[i][j] - lr * avgGrad;
                    w[i][j] += vw[i][j];
                }
            }
            for (int i = 0; i < b.length; i++) {
                float avgGrad = gb[i] * invBatch;
                vb[i] = momentum * vb[i] - lr * avgGrad;
                b[i] += vb[i];
            }
        }
    }

    public void zeroGradients(float[][][] gradWeights, float[][] gradBiases) {
        for (int l = 0; l < gradWeights.length; l++) {
            for (float[] row : gradWeights[l]) {
                Arrays.fill(row, 0.0f);
            }
            Arrays.fill(gradBiases[l], 0.0f);
        }
    }

    public float[][][] copyWeights() {
        float[][][] copy = new float[weights.length][][];
        for (int l = 0; l < weights.length; l++) {
            copy[l] = new float[weights[l].length][];
            for (int i = 0; i < weights[l].length; i++) {
                copy[l][i] = weights[l][i].clone();
            }
        }
        return copy;
    }

    public float[][] copyBiases() {
        float[][] copy = new float[biases.length][];
        for (int l = 0; l < biases.length; l++) {
            copy[l] = biases[l].clone();
        }
        return copy;
    }

    public void restoreWeights(float[][][] savedWeights, float[][] savedBiases) {
        for (int l = 0; l < weights.length; l++) {
            for (int i = 0; i < weights[l].length; i++) {
                System.arraycopy(savedWeights[l][i], 0, weights[l][i], 0, weights[l][i].length);
            }
            System.arraycopy(savedBiases[l], 0, biases[l], 0, biases[l].length);
        }
    }
}