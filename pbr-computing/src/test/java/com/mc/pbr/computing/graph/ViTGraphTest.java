package com.mc.pbr.computing.graph;

import org.junit.Test;
import static org.junit.Assert.*;

public class ViTGraphTest {
    static {
        String dllPath = System.getProperty("user.dir") + "/../pbr-opencl-native/target/classes/pbr_ocl.dll";
        try {
            System.load(dllPath);
            System.out.println("Loaded DLL from: " + dllPath);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load " + dllPath + ": " + e.getMessage());
            try {
                System.load("pbr_ocl.dll");
            } catch (UnsatisfiedLinkError ex) {
                System.err.println("Also failed to load from current directory.");
            }
        }
    }

    @Test
    public void testForward() {
        int embedDim = 64;
        int numLayers = 4;
        int numHeads = 4;
        int mlpDim = 128;
        int imageH = 32;
        int imageW = 32;
        int patchSize = 8;
        int inChannels = 4;
        int numClasses = 10;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, imageH, imageW, patchSize, inChannels, numClasses, seed, mppNumClasses);
        int batchSize = 2;
        int seqLen = (imageH / patchSize) * (imageW / patchSize);
        int featureDim = seqLen * inChannels;
        float[] input = new float[batchSize * featureDim];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        int finalH = imageH;
        int finalW = imageW;
        float[] output = new float[batchSize * finalH * finalW * numClasses];
        graph.forward(input, output, batchSize);
        assertEquals(batchSize * finalH * finalW * numClasses, output.length);
        for (float v : output) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }
        graph.close();
    }

    @Test
    public void testWeightsSerialization() {
        int embedDim = 64;
        int numLayers = 4;
        int numHeads = 4;
        int mlpDim = 128;
        int imageH = 32;
        int imageW = 32;
        int patchSize = 8;
        int inChannels = 4;
        int numClasses = 10;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, imageH, imageW, patchSize, inChannels, numClasses, seed, mppNumClasses);
        float[] weights = graph.getWeights();
        float[] biases = graph.getBiases();
        System.out.print("graph1 weights[0..9]: ");
        for (int i = 0; i < Math.min(10, weights.length); i++) {
            System.out.print(weights[i] + " ");
        }
        System.out.println();
        System.out.print("graph1 biases[0..9]: ");
        for (int i = 0; i < Math.min(10, biases.length); i++) {
            System.out.print(biases[i] + " ");
        }
        System.out.println();
        assertNotNull(weights);
        assertNotNull(biases);
        assertTrue(weights.length > 0);
        assertTrue(biases.length > 0);
        ViTGraph graph2 = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, imageH, imageW, patchSize, inChannels, numClasses, weights, biases, mppNumClasses);
        float[] weights2 = graph2.getWeights();
        System.out.print("graph2 weights[0..9]: ");
        for (int i = 0; i < Math.min(10, weights2.length); i++) {
            System.out.print(weights2[i] + " ");
        }
        System.out.println();
        int seqLen = (imageH / patchSize) * (imageW / patchSize);
        int featureDim = seqLen * inChannels;
        float[] input = new float[featureDim];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / input.length;
        }
        int finalH = imageH;
        int finalW = imageW;
        float[] out1 = new float[finalH * finalW * numClasses];
        float[] out2 = new float[finalH * finalW * numClasses];
        graph.forward(input, out1, 1);
        graph2.forward(input, out2, 1);
        for (int i = 0; i < out1.length; i++) {
            assertEquals(out1[i], out2[i], 1e-3f);
        }
        graph.close();
        graph2.close();
    }

    @Test
    public void testBackwardAndGradient() {
        int embedDim = 32;
        int numLayers = 2;
        int numHeads = 4;
        int mlpDim = 64;
        int imageH = 16;
        int imageW = 16;
        int patchSize = 8;
        int inChannels = 5;
        int numClasses = 3;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, imageH, imageW, patchSize, inChannels, numClasses, seed, mppNumClasses);
        int batchSize = 2;
        int seqLen = (imageH / patchSize) * (imageW / patchSize);
        int featureDim = seqLen * inChannels;
        float[] input = new float[batchSize * featureDim];
        int finalH = imageH;
        int finalW = imageW;
        float[] label = new float[batchSize * finalH * finalW * numClasses];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) (i + 1) / input.length;
        }
        for (int i = 0; i < label.length; i++) {
            label[i] = (float) i / label.length;
        }
        float[] output = new float[batchSize * finalH * finalW * numClasses];
        graph.forward(input, output, batchSize);

        float[] gradOutput = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            gradOutput[i] = 2.0f * (output[i] - label[i]) / (output.length * batchSize);
        }

        graph.zeroGradients();
        graph.backward(input, label, gradOutput, batchSize);

        float[] weightsOrig = graph.getWeights();
        float[] gradNum = new float[weightsOrig.length];
        float epsilon = 1e-3f;
        ViTGraph tempGraph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, imageH, imageW, patchSize, inChannels, numClasses, weightsOrig.clone(), graph.getBiases().clone(), mppNumClasses);
        float[] wTemp = weightsOrig.clone();
        int total = weightsOrig.length;
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < total; i++) {
            float originalVal = wTemp[i];
            wTemp[i] = originalVal + epsilon;
            tempGraph.setWeights(wTemp);
            float[] outPlus = new float[batchSize * finalH * finalW * numClasses];
            tempGraph.forward(input, outPlus, batchSize);
            double lossPlus = 0.0;
            for (int b = 0; b < batchSize; b++) {
                for (int p = 0; p < finalH * finalW * numClasses; p++) {
                    int idx = b * (finalH * finalW * numClasses) + p;
                    double diff = (double)outPlus[idx] - (double)label[idx];
                    lossPlus += diff * diff;
                }
            }
            lossPlus /= (batchSize * finalH * finalW * numClasses);

            wTemp[i] = originalVal - epsilon;
            tempGraph.setWeights(wTemp);
            float[] outMinus = new float[batchSize * finalH * finalW * numClasses];
            tempGraph.forward(input, outMinus, batchSize);
            double lossMinus = 0.0;
            for (int b = 0; b < batchSize; b++) {
                for (int p = 0; p < finalH * finalW * numClasses; p++) {
                    int idx = b * (finalH * finalW * numClasses) + p;
                    double diff = (double)outMinus[idx] - (double)label[idx];
                    lossMinus += diff * diff;
                }
            }
            lossMinus /= (batchSize * finalH * finalW * numClasses);

            wTemp[i] = originalVal;
            gradNum[i] = (float)((lossPlus - lossMinus) / (2.0 * epsilon));

            long elapsed = System.currentTimeMillis() - startTime;
            double pct = (double) (i + 1) / total * 100.0;
            double eta = (i + 1) > 0 ? (elapsed / (double) (i + 1)) * (total - i - 1) / 1000.0 : 0.0;
            System.out.printf("\r[Gradient Check] %5.1f%% (%d/%d) | Elapsed: %ds | ETA: %.1fs", pct, i + 1, total, elapsed / 1000, eta);
            System.out.flush();
        }
        System.out.println();
        tempGraph.close();

        graph.zeroGradients();
        graph.backward(input, label, gradOutput, batchSize);
        float[] gradGPU = graph.getGradients();

        int mismatchCount = 0;
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        StringBuilder details = new StringBuilder();

        for (int i = 0; i < gradGPU.length; i++) {
            float absDiff = Math.abs(gradGPU[i] - gradNum[i]);
            float denom = Math.max(Math.abs(gradGPU[i]), Math.abs(gradNum[i]));
            float relDiff = denom > 1e-7f ? absDiff / denom : absDiff;

            if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
            if (relDiff > maxRelDiff) maxRelDiff = relDiff;

            boolean fail;
            if (Math.abs(gradGPU[i]) < 1e-7f) {
                fail = false;
            } else {
                fail = absDiff > 1e-3f && relDiff > 1e-2f;
            }
            if (fail) {
                mismatchCount++;
                if (mismatchCount <= 10) {
                    details.append(String.format(
                            "  idx=%d | GPU=%.6f | NUM=%.6f | absDiff=%.6f | relDiff=%.4f\n",
                            i, gradGPU[i], gradNum[i], absDiff, relDiff));
                }
            }
        }

        System.out.printf("Total weights: %d | Mismatches: %d | MaxAbsDiff: %.6f | MaxRelDiff: %.4f\n",
                gradGPU.length, mismatchCount, maxAbsDiff, maxRelDiff);
        if (mismatchCount > 0) {
            System.out.println("First mismatches:");
            System.out.print(details.toString());
        }

        assertEquals("Gradient check failed: " + mismatchCount + "/" + gradGPU.length + " mismatches",
                0, mismatchCount);

        graph.close();
    }
}