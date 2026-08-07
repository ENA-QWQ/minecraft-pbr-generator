package com.mc.pbr.computing;

import com.mc.pbr.training.TinyMLP;
import java.util.Random;

public class CPUComputingBackend implements ComputingBackend {
    private final TinyMLP mlp;
    private final int featureDim;
    private final int labelDim;
    private final int[] layerSizes;
    private final float[][][] gradWeights;
    private final float[][] gradBiases;
    private final float[][] activations;
    private final float[][] zs;
    private final float[][] deltas;

    public CPUComputingBackend(int[] layerSizes, long seed) {
        this.layerSizes = layerSizes.clone();
        this.featureDim = layerSizes[0];
        this.labelDim = layerSizes[layerSizes.length - 1];
        this.mlp = new TinyMLP(layerSizes, new Random(seed));

        int numLayers = layerSizes.length - 1;
        gradWeights = new float[numLayers][][];
        gradBiases = new float[numLayers][];
        activations = new float[layerSizes.length][];
        zs = new float[numLayers][];
        deltas = new float[numLayers][];

        for (int l = 0; l < numLayers; l++) {
            int outDim = layerSizes[l + 1];
            int inDim = layerSizes[l];
            gradWeights[l] = new float[outDim][inDim];
            gradBiases[l] = new float[outDim];
            zs[l] = new float[outDim];
            deltas[l] = new float[outDim];
        }
        for (int l = 0; l < layerSizes.length; l++) {
            activations[l] = new float[layerSizes[l]];
        }
    }

    public CPUComputingBackend(int[] layerSizes, float[] weights, float[] biases) {
        this(layerSizes, 0);
        if (weights != null) setWeights(weights);
        if (biases != null) setBiases(biases);
    }

    @Override
    public int getFeatureDim() { return featureDim; }

    @Override
    public int getLabelDim() { return labelDim; }

    @Override
    public int[] getLayerSizes() { return layerSizes.clone(); }

    @Override
    public void forward(float[] input, float[] output) {
        float[][] acts = new float[layerSizes.length][];
        float[][] zsLocal = new float[layerSizes.length - 1][];
        acts[0] = input;
        for (int l = 1; l < layerSizes.length; l++) {
            acts[l] = new float[layerSizes[l]];
        }
        for (int l = 0; l < layerSizes.length - 1; l++) {
            zsLocal[l] = new float[layerSizes[l + 1]];
        }
        mlp.forward(input, acts, zsLocal);
        float[] last = acts[layerSizes.length - 1];
        System.arraycopy(last, 0, output, 0, last.length);
    }

    @Override
    public void forwardBatch(float[] input, float[] output, int batchSize) {
        float[] singleInput = new float[featureDim];
        float[] singleOutput = new float[labelDim];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(input, b * featureDim, singleInput, 0, featureDim);
            forward(singleInput, singleOutput);
            System.arraycopy(singleOutput, 0, output, b * labelDim, labelDim);
        }
    }

    @Override
    public void backward(float[] input, float[] label, float[] gradOutput) {
        throw new UnsupportedOperationException("Use backwardBatch for training");
    }

    @Override
    public void backwardBatch(float[] input, float[] label, float[] gradOutput, int batchSize) {
        zeroGradients();
        float[] singleInput = new float[featureDim];
        float[] singleLabel = new float[labelDim];
        for (int b = 0; b < batchSize; b++) {
            System.arraycopy(input, b * featureDim, singleInput, 0, featureDim);
            System.arraycopy(label, b * labelDim, singleLabel, 0, labelDim);
            // forward pass to fill activations and zs
            System.arraycopy(singleInput, 0, activations[0], 0, featureDim);
            mlp.forward(singleInput, activations, zs);
            // backward accumulates gradients
            mlp.backward(singleLabel, activations, zs, deltas, gradWeights, gradBiases);
        }
    }

    @Override
    public void update(float[][] gradWeightsArg, float[] gradBiasesArg, int batchSize, float lr, float momentum) {
        mlp.update(gradWeights, gradBiases, batchSize, lr, momentum);
    }

    @Override
    public void zeroGradients() {
        int numLayers = layerSizes.length - 1;
        for (int l = 0; l < numLayers; l++) {
            for (int i = 0; i < gradWeights[l].length; i++) {
                java.util.Arrays.fill(gradWeights[l][i], 0.0f);
            }
            java.util.Arrays.fill(gradBiases[l], 0.0f);
        }
    }

    @Override
    public float[] getWeights() {
        float[][][] w = mlp.getWeights();
        int total = 0;
        for (int l = 0; l < w.length; l++) {
            for (int i = 0; i < w[l].length; i++) {
                total += w[l][i].length;
            }
        }
        float[] result = new float[total];
        int offset = 0;
        for (int l = 0; l < w.length; l++) {
            for (int i = 0; i < w[l].length; i++) {
                System.arraycopy(w[l][i], 0, result, offset, w[l][i].length);
                offset += w[l][i].length;
            }
        }
        return result;
    }

    @Override
    public float[] getBiases() {
        float[][] b = mlp.getBiases();
        int total = 0;
        for (int l = 0; l < b.length; l++) {
            total += b[l].length;
        }
        float[] result = new float[total];
        int offset = 0;
        for (int l = 0; l < b.length; l++) {
            System.arraycopy(b[l], 0, result, offset, b[l].length);
            offset += b[l].length;
        }
        return result;
    }

    @Override
    public void setWeights(float[] weights) {
        int numLayers = layerSizes.length - 1;
        int offset = 0;
        float[][][] w = mlp.getWeights();
        for (int l = 0; l < numLayers; l++) {
            int rows = layerSizes[l + 1];
            int cols = layerSizes[l];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(weights, offset, w[l][i], 0, cols);
                offset += cols;
            }
        }
    }

    @Override
    public void setBiases(float[] biases) {
        int numLayers = layerSizes.length - 1;
        int offset = 0;
        float[][] b = mlp.getBiases();
        for (int l = 0; l < numLayers; l++) {
            int rows = layerSizes[l + 1];
            System.arraycopy(biases, offset, b[l], 0, rows);
            offset += rows;
        }
    }

    @Override
    public void close() {}
}