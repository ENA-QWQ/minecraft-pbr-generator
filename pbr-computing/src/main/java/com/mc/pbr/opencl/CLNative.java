package com.mc.pbr.opencl;

public class CLNative {
    static {
        try {
            System.loadLibrary("pbr_ocl");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[WARN] Failed to load pbr_ocl native library: " + e.getMessage());
        }
    }

    public static native long create(int[] layerSizes, long seed);
    public static native long createWithWeights(int[] layerSizes, float[] weights, float[] biases);
    public static native void destroy(long handle);
    public static native void forward(long handle, float[] input, float[] output, int batchSize);
    public static native void backward(long handle, float[] input, float[] label, float[] gradOutput, int batchSize);
    public static native void update(long handle, float[] gradWeights, float[] gradBiases, int batchSize, float lr, float momentum);
    public static native void zeroGradients(long handle);
    public static native float[] getWeights(long handle);
    public static native float[] getBiases(long handle);
    public static native void setWeights(long handle, float[] weights);
    public static native void setBiases(long handle, float[] biases);
}