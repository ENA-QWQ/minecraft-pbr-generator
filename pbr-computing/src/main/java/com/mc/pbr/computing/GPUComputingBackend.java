package com.mc.pbr.computing;

import com.mc.pbr.computing.graph.ModelGraph;
import com.mc.pbr.computing.graph.MLPGraph;

public class GPUComputingBackend implements ComputingBackend {
    private final ModelGraph graph;
    private final int featureDim;
    private final int labelDim;
    private final int[] layerSizes;

    public GPUComputingBackend(int[] layerSizes, long seed) {
        this.graph = new MLPGraph(layerSizes, seed);
        this.layerSizes = graph.getLayerSizes();
        this.featureDim = graph.getFeatureDim();
        this.labelDim = graph.getLabelDim();
    }

    public GPUComputingBackend(int[] layerSizes, float[] weights, float[] biases) {
        this.graph = new MLPGraph(layerSizes, weights, biases);
        this.layerSizes = graph.getLayerSizes();
        this.featureDim = graph.getFeatureDim();
        this.labelDim = graph.getLabelDim();
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
    public void forward(float[] input, float[] output) {
        graph.forward(input, output, 1);
    }

    @Override
    public void forwardBatch(float[] input, float[] output, int batchSize) {
        graph.forward(input, output, batchSize);
    }

    @Override
    public void backward(float[] input, float[] label, float[] gradOutput) {
        graph.backward(input, label, gradOutput, 1);
    }

    @Override
    public void backwardBatch(float[] input, float[] label, float[] gradOutput, int batchSize) {
        graph.backward(input, label, gradOutput, batchSize);
    }

    @Override
    public void update(float[][] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum) {
        graph.update(gradWeights, gradBiases, batchSize, lr, momentum);
    }

    @Override
    public void zeroGradients() {
        graph.zeroGradients();
    }

    @Override
    public float[] getWeights() {
        return graph.getWeights();
    }

    @Override
    public float[] getBiases() {
        return graph.getBiases();
    }

    @Override
    public void setWeights(float[] weights) {
        graph.setWeights(weights);
    }

    @Override
    public void setBiases(float[] biases) {
        graph.setBiases(biases);
    }

    @Override
    public void close() {
        graph.close();
    }
}