package com.mc.pbr.computing.graph;

import com.mc.pbr.opencl.CLNative;

public class MLPGraph implements ModelGraph {
    private final long nativeHandle;
    private final int featureDim;
    private final int labelDim;
    private final int[] layerSizes;
    private boolean closed = false;

    public MLPGraph(int[] layerSizes, long seed) {
        this.layerSizes = layerSizes.clone();
        this.featureDim = layerSizes[0];
        this.labelDim = layerSizes[layerSizes.length - 1];
        this.nativeHandle = CLNative.create(layerSizes, seed);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize MLP graph (OpenCL init failed)");
        }
    }

    public MLPGraph(int[] layerSizes, float[] weights, float[] biases) {
        this.layerSizes = layerSizes.clone();
        this.featureDim = layerSizes[0];
        this.labelDim = layerSizes[layerSizes.length - 1];
        this.nativeHandle = CLNative.createWithWeights(layerSizes, weights, biases);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize MLP graph with weights (OpenCL init failed)");
        }
    }

    @Override
    public int getFeatureDim() {
        return featureDim;
    }

    @Override
    public int getLabelDim() {
        return labelDim;
    }

    @Override
    public int[] getLayerSizes() {
        return layerSizes.clone();
    }

    @Override
    public void forward(float[] input, float[] output, int batchSize) {
        checkClosed();
        CLNative.forward(nativeHandle, input, output, batchSize);
    }

    @Override
    public void backward(float[] input, float[] label, float[] gradOutput, int batchSize) {
        checkClosed();
        CLNative.backward(nativeHandle, input, label, gradOutput, batchSize);
    }

    @Override
    public void update(float[][] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum) {
        checkClosed();
        float[] flatGradWeights = null;
        if (gradWeights != null) {
            flatGradWeights = flattenGradWeights(gradWeights);
        }
        CLNative.update(nativeHandle, flatGradWeights, gradBiases, batchSize, lr, momentum);
    }

    private float[] flattenGradWeights(float[][] gradWeights) {
        int total = 0;
        for (float[] row : gradWeights) total += row.length;
        float[] flat = new float[total];
        int offset = 0;
        for (float[] row : gradWeights) {
            System.arraycopy(row, 0, flat, offset, row.length);
            offset += row.length;
        }
        return flat;
    }

    @Override
    public void zeroGradients() {
        checkClosed();
        CLNative.zeroGradients(nativeHandle);
    }

    @Override
    public float[] getWeights() {
        checkClosed();
        return CLNative.getWeights(nativeHandle);
    }

    @Override
    public float[] getBiases() {
        checkClosed();
        return CLNative.getBiases(nativeHandle);
    }

    @Override
    public void setWeights(float[] weights) {
        checkClosed();
        CLNative.setWeights(nativeHandle, weights);
    }

    @Override
    public void setBiases(float[] biases) {
        checkClosed();
        CLNative.setBiases(nativeHandle, biases);
    }

    private void checkClosed() {
        if (closed) throw new IllegalStateException("MLP graph already closed");
    }

    @Override
    public void close() {
        if (!closed) {
            CLNative.destroy(nativeHandle);
            closed = true;
        }
    }
}