package com.mc.pbr.training;

import java.io.Serializable;

public class ModelData implements Serializable {
    private static final long serialVersionUID = 1L;
    public final String modelType;
    public final int[] layerSizes;
    public final float[] weights;
    public final float[] biases;
    public final int seqLen;
    public final int embedDim;
    public final int numLayers;
    public final int numHeads;
    public final int mlpDim;
    public final int inChannels;
    public final int mppNumClasses;

    public ModelData(String modelType, int[] layerSizes, float[] weights, float[] biases,
                     int seqLen, int embedDim, int numLayers, int numHeads, int mlpDim, int inChannels, int mppNumClasses) {
        this.modelType = modelType;
        this.layerSizes = layerSizes;
        this.weights = weights;
        this.biases = biases;
        this.seqLen = seqLen;
        this.embedDim = embedDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.mlpDim = mlpDim;
        this.inChannels = inChannels;
        this.mppNumClasses = mppNumClasses;
    }
}