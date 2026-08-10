package com.mc.pbr.computing.graph;

import com.mc.pbr.opencl.CLNative;

public class ViTGraph implements ModelGraph {
    private final long nativeHandle;
    private final int featureDim;
    private final int labelDim;
    private final int[] layerSizes;
    private boolean closed = false;

    public ViTGraph(int embedDim, int numLayers, int numHeads, int mlpDim, int seqLen, int inChannels, long seed) {
        this.nativeHandle = CLNative.createViT(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, seed);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize ViT graph");
        }
        this.featureDim = seqLen * inChannels;
        this.labelDim = 1;
        this.layerSizes = new int[]{featureDim, embedDim, numLayers, numHeads, mlpDim};
    }

    public ViTGraph(int embedDim, int numLayers, int numHeads, int mlpDim, int seqLen, int inChannels, float[] weights, float[] biases) {
        int totalWeights = 0;
        int totalBiases = 0;
        totalWeights += inChannels * embedDim;
        totalWeights += embedDim;
        totalWeights += (seqLen + 1) * embedDim;
        for (int l = 0; l < numLayers; l++) {
            totalWeights += embedDim;
            totalBiases += embedDim;
            totalWeights += embedDim * (3 * embedDim);
            totalBiases += 3 * embedDim;
            totalWeights += embedDim * embedDim;
            totalBiases += embedDim;
            totalWeights += embedDim;
            totalBiases += embedDim;
            totalWeights += embedDim * mlpDim;
            totalBiases += mlpDim;
            totalWeights += mlpDim * embedDim;
            totalBiases += embedDim;
        }
        totalWeights += embedDim;
        totalBiases += embedDim;
        totalWeights += embedDim;
        totalBiases += 1;

        if (weights != null && weights.length != totalWeights) {
            throw new RuntimeException("Invalid weights length: expected " + totalWeights + ", got " + weights.length);
        }
        if (biases != null && biases.length != totalBiases) {
            throw new RuntimeException("Invalid biases length: expected " + totalBiases + ", got " + biases.length);
        }
        this.nativeHandle = CLNative.createViTWithWeights(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, weights, biases);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize ViT graph with weights");
        }
        this.featureDim = seqLen * inChannels;
        this.labelDim = 1;
        this.layerSizes = new int[]{featureDim, embedDim, numLayers, numHeads, mlpDim};
    }

    @Override
    public int getFeatureDim() { return featureDim; }
    @Override
    public int getLabelDim() { return labelDim; }
    @Override
    public int[] getLayerSizes() { return layerSizes.clone(); }

    @Override
    public void forward(float[] input, float[] output, int batchSize) {
        checkClosed();
        CLNative.forwardViT(nativeHandle, input, output, batchSize);
    }

    @Override
    public void backward(float[] input, float[] label, float[] gradOutput, int batchSize) {
        throw new UnsupportedOperationException("Backward not implemented");
    }

    @Override
    public void update(float[][] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum) {
        throw new UnsupportedOperationException("Update not implemented");
    }

    @Override
    public void zeroGradients() {
        throw new UnsupportedOperationException("zeroGradients not implemented");
    }

    @Override
    public float[] getWeights() {
        checkClosed();
        return CLNative.getViTWeights(nativeHandle);
    }

    @Override
    public float[] getBiases() {
        checkClosed();
        return CLNative.getViTBiases(nativeHandle);
    }

    @Override
    public void setWeights(float[] weights) {
        checkClosed();
        CLNative.setViTWeights(nativeHandle, weights);
    }

    @Override
    public void setBiases(float[] biases) {
        checkClosed();
        CLNative.setViTBiases(nativeHandle, biases);
    }

    private void checkClosed() {
        if (closed) throw new IllegalStateException("ViT graph closed");
    }

    @Override
    public void close() {
        if (!closed) {
            CLNative.destroyViT(nativeHandle);
            closed = true;
        }
    }
}