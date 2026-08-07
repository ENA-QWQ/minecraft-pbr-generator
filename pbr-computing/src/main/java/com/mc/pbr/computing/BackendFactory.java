package com.mc.pbr.computing;

public class BackendFactory {
    public static ComputingBackend create(String backendType, int[] layerSizes, long seed) {
        if ("gpu".equalsIgnoreCase(backendType)) {
            try {
                return new GPUComputingBackend(layerSizes, seed);
            } catch (Exception e) {
                System.err.println("[WARN] GPU backend unavailable, falling back to CPU: " + e.getMessage());
                return new CPUComputingBackend(layerSizes, seed);
            }
        }
        return new CPUComputingBackend(layerSizes, seed);
    }

    public static ComputingBackend createFromWeights(String backendType, int[] layerSizes,
                                                     float[] weights, float[] biases) {
        if ("gpu".equalsIgnoreCase(backendType)) {
            try {
                return new GPUComputingBackend(layerSizes, weights, biases);
            } catch (Exception e) {
                System.err.println("[WARN] GPU backend unavailable, falling back to CPU: " + e.getMessage());
                return new CPUComputingBackend(layerSizes, weights, biases);
            }
        }
        return new CPUComputingBackend(layerSizes, weights, biases);
    }
}