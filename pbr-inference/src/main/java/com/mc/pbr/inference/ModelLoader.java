package com.mc.pbr.inference;

import com.mc.pbr.training.TinyMLP;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;

public class ModelLoader {
    private final TinyMLP mlp;
    private final int numLayers;

    private final float[][] activations;
    private final float[][] zs;
    private final double[] outputBuffer;

    public ModelLoader(String modelPath) throws Exception {
        try (FileInputStream fis = new FileInputStream(modelPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            Object obj = ois.readObject();
            if (!(obj instanceof TinyMLP)) {
                throw new RuntimeException("Deserialization failed, model is not TinyMLP type! Current type: " + obj.getClass().getName());
            }
            this.mlp = (TinyMLP) obj;
        }

        int[] layerSizes = mlp.getLayerSizes();
        this.numLayers = layerSizes.length - 1;

        this.activations = new float[numLayers + 1][];
        this.zs = new float[numLayers][];

        for (int l = 1; l <= numLayers; l++) {
            activations[l] = new float[layerSizes[l]];
        }
        for (int l = 0; l < numLayers; l++) {
            zs[l] = new float[layerSizes[l + 1]];
        }

        this.outputBuffer = new double[layerSizes[numLayers]];

        System.out.println("[INFO] Loading TinyMLP, layer structure: " + Arrays.toString(layerSizes));
    }

    public double[] predict(float[] featuresFloat, double[] featuresDouble) {

        mlp.forward(featuresFloat, activations, zs);

        float[] lastActivation = activations[numLayers];
        for (int i = 0; i < lastActivation.length; i++) {
            outputBuffer[i] = lastActivation[i];
        }

        return outputBuffer;
    }
}