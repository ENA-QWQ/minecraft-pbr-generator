package com.mc.pbr.inference;

import com.mc.pbr.training.ModelData;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class ModelLoader {
    private final int[] layerSizes;
    private final float[] weights;
    private final float[] biases;

    public ModelLoader(String modelPath) throws Exception {
        try (FileInputStream fis = new FileInputStream(modelPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            Object obj = ois.readObject();
            if (obj instanceof ModelData) {
                ModelData data = (ModelData) obj;
                this.layerSizes = data.layerSizes;
                this.weights = data.weights;
                this.biases = data.biases;
            } else {
                throw new RuntimeException("Unsupported model format: " + obj.getClass().getName());
            }
        }
    }

    public int[] getLayerSizes() { return layerSizes; }
    public float[] getWeights() { return weights; }
    public float[] getBiases() { return biases; }
}