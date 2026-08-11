package com.mc.pbr.opencl;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class CLNative {
    private static boolean loaded = false;

    static {
        if (!loaded) {
            String libName = "pbr_ocl.dll";
            InputStream in = CLNative.class.getResourceAsStream("/" + libName);
            boolean success = false;
            if (in != null) {
                try {
                    File tempFile = File.createTempFile("pbr_ocl", ".dll");
                    tempFile.deleteOnExit();
                    try (OutputStream out = new FileOutputStream(tempFile)) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = in.read(buffer)) != -1) {
                            out.write(buffer, 0, len);
                        }
                    }
                    System.load(tempFile.getAbsolutePath());
                    success = true;
                } catch (Exception e) {
                    System.err.println("[ERROR] Failed to extract pbr_ocl from JAR: " + e.getMessage());
                }
            }
            if (!success) {
                try {
                    System.loadLibrary("pbr_ocl");
                    success = true;
                } catch (UnsatisfiedLinkError e) {
                    System.err.println("[ERROR] Failed to load pbr_ocl native library: " + e.getMessage());
                }
            }
            loaded = success;
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

    public static native long createViT(int embedDim, int numLayers, int numHeads, int mlpDim, int seqLen, int inChannels, long seed);
    public static native long createViTWithWeights(int embedDim, int numLayers, int numHeads, int mlpDim, int seqLen, int inChannels, float[] weights, float[] biases);
    public static native void destroyViT(long handle);
    public static native void forwardViT(long handle, float[] input, float[] output, int batchSize);
    public static native void backwardViT(long handle, float[] input, float[] label, float[] gradOutput, int batchSize);
    public static native void adamwUpdateViT(long handle, int batchSize, float lr, float beta1, float beta2, float epsilon, float weightDecay, int step);
    public static native void clipGradientsViT(long handle, float maxNorm);
    public static native void zeroGradientsViT(long handle);
    public static native float[] getViTWeights(long handle);
    public static native float[] getViTBiases(long handle);
    public static native void setViTWeights(long handle, float[] weights);
    public static native void setViTBiases(long handle, float[] biases);
    public static native float mppForwardViT(long handle, int[] maskIndices, int[] targets, int batchSize, int numMasked, int numClasses);
    public static native void mppBackwardViT(long handle, int[] maskIndices, int[] targets, int batchSize, int numMasked, int numClasses, float lossScale);
    public static native float[] getViTGradients(long handle);
}