package com.mc.pbr.inference;

import com.mc.pbr.training.ModelData;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class ModelLoader {
    private final String modelType;
    private final int[] layerSizes;
    private final float[] weights;
    private final float[] biases;
    private final int seqLen;
    private final int embedDim;
    private final int numLayers;
    private final int numHeads;
    private final int mlpDim;
    private final int inChannels;
    private final int mppNumClasses;

    public ModelLoader(String modelPath) throws Exception {
        try (FileInputStream fis = new FileInputStream(modelPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            Object obj = ois.readObject();
            if (obj instanceof ModelData) {
                ModelData data = (ModelData) obj;
                this.modelType = data.modelType;
                this.layerSizes = data.layerSizes;
                this.weights = data.weights;
                this.biases = data.biases;
                this.seqLen = data.seqLen;
                this.embedDim = data.embedDim;
                this.numLayers = data.numLayers;
                this.numHeads = data.numHeads;
                this.mlpDim = data.mlpDim;
                this.inChannels = data.inChannels;
                this.mppNumClasses = data.mppNumClasses;
            } else {
                throw new RuntimeException("Unsupported model format: " + obj.getClass().getName());
            }
        }
    }

    public String getModelType() { return modelType; }
    public int[] getLayerSizes() { return layerSizes; }
    public float[] getWeights() { return weights; }
    public float[] getBiases() { return biases; }
    public int getSeqLen() { return seqLen; }
    public int getEmbedDim() { return embedDim; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return numHeads; }
    public int getMlpDim() { return mlpDim; }
    public int getInChannels() { return inChannels; }
    public int getMppNumClasses() { return mppNumClasses; }
}