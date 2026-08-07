package com.mc.pbr.computing;

public interface ComputingBackend {
    int getFeatureDim();
    int getLabelDim();
    int[] getLayerSizes();
    void forward(float[] input, float[] output);
    void forwardBatch(float[] input, float[] output, int batchSize);
    void backward(float[] input, float[] label, float[] gradOutput);
    void backwardBatch(float[] input, float[] label, float[] gradOutput, int batchSize);
    void update(float[][] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum);
    void zeroGradients();
    float[] getWeights();
    float[] getBiases();
    void setWeights(float[] weights);
    void setBiases(float[] biases);
    void close();
}