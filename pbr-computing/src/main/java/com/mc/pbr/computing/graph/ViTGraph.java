package com.mc.pbr.computing.graph;

import com.mc.pbr.opencl.CLNative;

public class ViTGraph implements ModelGraph {
    private final long nativeHandle;
    private final int featureDim;
    private final int labelDim;
    private final int[] layerSizes;
    private final int seqLen;
    private final int patchH;
    private final int patchW;
    private final int numClasses;
    private final int imageH;
    private final int imageW;
    private boolean closed = false;

    public ViTGraph(int embedDim, int numLayers, int numHeads, int mlpDim, int imageH, int imageW, int patchSize, int inChannels, int numClasses, long seed, int mppNumClasses) {
        this.imageH = imageH;
        this.imageW = imageW;
        this.patchH = imageH / patchSize;
        this.patchW = imageW / patchSize;
        this.seqLen = this.patchH * this.patchW;
        this.numClasses = numClasses;
        this.nativeHandle = CLNative.createViT(embedDim, numLayers, numHeads, mlpDim, this.seqLen, inChannels, this.patchH, this.patchW, numClasses, seed, mppNumClasses);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize ViT graph");
        }
        this.featureDim = this.seqLen * inChannels;
        this.labelDim = imageH * imageW;
        this.layerSizes = new int[]{featureDim, embedDim, numLayers, numHeads, mlpDim};
    }

    public ViTGraph(int embedDim, int numLayers, int numHeads, int mlpDim, int imageH, int imageW, int patchSize, int inChannels, int numClasses, float[] weights, float[] biases, int mppNumClasses) {
        this.imageH = imageH;
        this.imageW = imageW;
        this.patchH = imageH / patchSize;
        this.patchW = imageW / patchSize;
        this.seqLen = this.patchH * this.patchW;
        this.numClasses = numClasses;
        int totalWeights = 0;
        int totalBiases = 0;
        totalWeights += inChannels * embedDim;
        totalWeights += embedDim;
        totalWeights += (this.seqLen + 1) * embedDim;
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
        int decIn = embedDim;
        for (int i = 0; i < 3; i++) {
            int decOut = (i == 0) ? embedDim / 2 : (i == 1) ? embedDim / 4 : numClasses;
            totalWeights += decOut * decIn * 9;
            totalBiases += decOut;
            decIn = decOut;
        }
        if (weights != null && weights.length != totalWeights) {
            throw new RuntimeException("Invalid weights length: expected " + totalWeights + ", got " + weights.length);
        }
        if (biases != null && biases.length != totalBiases) {
            throw new RuntimeException("Invalid biases length: expected " + totalBiases + ", got " + biases.length);
        }
        this.nativeHandle = CLNative.createViTWithWeights(embedDim, numLayers, numHeads, mlpDim, this.seqLen, inChannels, this.patchH, this.patchW, numClasses, weights, biases, mppNumClasses);
        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize ViT graph with weights");
        }
        this.featureDim = this.seqLen * inChannels;
        this.labelDim = imageH * imageW;
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
        checkClosed();
        CLNative.backwardViT(nativeHandle, input, label, gradOutput, batchSize);
    }

    @Override
    public void update(float[][] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum) {
        checkClosed();
        CLNative.adamwUpdateViT(nativeHandle, batchSize, lr, 0.9f, 0.999f, 1e-8f, 0.1f, 0);
    }

    public void adamwUpdate(int batchSize, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step) {
        checkClosed();
        CLNative.adamwUpdateViT(nativeHandle, batchSize, lr, beta1, beta2, epsilon, weightDecay, step);
    }

    public void clipGradients(float maxNorm) {
        checkClosed();
        CLNative.clipGradientsViT(nativeHandle, maxNorm);
    }

    @Override
    public void zeroGradients() {
        checkClosed();
        CLNative.zeroGradientsViT(nativeHandle);
    }

    @Override
    public float[] getWeights() {
        checkClosed();
        return CLNative.getViTWeights(nativeHandle);
    }

    public float[] getGradients() {
        checkClosed();
        return CLNative.getViTGradients(nativeHandle);
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

    public float mppForward(int[] maskIndices, int[] targets, int batchSize, int numMasked, int numClasses) {
        checkClosed();
        return CLNative.mppForwardViT(nativeHandle, maskIndices, targets, batchSize, numMasked, numClasses);
    }

    public void mppBackward(int[] maskIndices, int[] targets, int batchSize, int numMasked, int numClasses, float lossScale) {
        checkClosed();
        CLNative.mppBackwardViT(nativeHandle, maskIndices, targets, batchSize, numMasked, numClasses, lossScale);
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