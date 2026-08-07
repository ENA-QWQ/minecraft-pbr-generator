package com.mc.pbr.training;

import java.io.Serializable;

public class ModelData implements Serializable {
    private static final long serialVersionUID = 1L;
    public final int[] layerSizes;
    public final float[] weights;
    public final float[] biases;

    public ModelData(int[] layerSizes, float[] weights, float[] biases) {
        this.layerSizes = layerSizes;
        this.weights = weights;
        this.biases = biases;
    }
}