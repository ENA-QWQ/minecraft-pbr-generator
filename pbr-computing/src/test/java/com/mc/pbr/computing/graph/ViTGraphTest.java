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
        int seqLen = 7;
        int inChannels = 4;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, seed, mppNumClasses);
        int batchSize = 2;
        int featureDim = seqLen * inChannels;
        float[] input = new float[batchSize * featureDim];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        float[] output = new float[batchSize];
        graph.forward(input, output, batchSize);
        assertEquals(batchSize, output.length);
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
        int seqLen = 7;
        int inChannels = 4;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, seed, mppNumClasses);
        float[] weights = graph.getWeights();
        float[] biases = graph.getBiases();
        assertNotNull(weights);
        assertNotNull(biases);
        assertTrue(weights.length > 0);
        assertTrue(biases.length > 0);
        ViTGraph graph2 = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, weights, biases, mppNumClasses);
        float[] input = new float[seqLen * inChannels];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) i / input.length;
        }
        float[] out1 = new float[1];
        float[] out2 = new float[1];
        graph.forward(input, out1, 1);
        graph2.forward(input, out2, 1);
        assertEquals(out1[0], out2[0], 1e-6f);
        graph.close();
        graph2.close();
    }

    @Test
    public void testBackwardAndGradient() {
        int embedDim = 32;
        int numLayers = 2;
        int numHeads = 4;
        int mlpDim = 64;
        int seqLen = 4;
        int inChannels = 5;
        long seed = 42L;
        int mppNumClasses = 512;
        ViTGraph graph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, seed, mppNumClasses);
        int batchSize = 2;
        int featureDim = seqLen * inChannels;
        float[] input = new float[batchSize * featureDim];
        float[] label = new float[batchSize];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) (i + 1) / input.length;
        }
        for (int i = 0; i < batchSize; i++) {
            label[i] = (float) i / batchSize;
        }
        float[] output = new float[batchSize];
        graph.forward(input, output, batchSize);

        // MSE Loss = sum((y-yhat)^2) / N
        // dLoss/dOutput = 2 * (yhat - y) / N
        float[] gradOutput = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            gradOutput[i] = 2.0f * (output[i] - label[i]) / batchSize;
        }

        graph.zeroGradients();
        graph.backward(input, label, gradOutput, batchSize);

        float[] weightsOrig = graph.getWeights();
        float[] gradNum = new float[weightsOrig.length];
        float epsilon = 1e-3f;
        ViTGraph tempGraph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, weightsOrig.clone(), graph.getBiases().clone(), mppNumClasses);
        float[] wTemp = weightsOrig.clone();
        int total = weightsOrig.length;
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < total; i++) {
            float originalVal = wTemp[i];
            wTemp[i] = originalVal + epsilon;
            tempGraph.setWeights(wTemp);
            float[] outPlus = new float[batchSize];
            tempGraph.forward(input, outPlus, batchSize);
            double lossPlus = 0.0;
            for (int b = 0; b < batchSize; b++) {
                double diff = (double)outPlus[b] - (double)label[b];
                lossPlus += diff * diff;
            }
            lossPlus /= batchSize;

            wTemp[i] = originalVal - epsilon;
            tempGraph.setWeights(wTemp);
            float[] outMinus = new float[batchSize];
            tempGraph.forward(input, outMinus, batchSize);
            double lossMinus = 0.0;
            for (int b = 0; b < batchSize; b++) {
                double diff = (double)outMinus[b] - (double)label[b];
                lossMinus += diff * diff;
            }
            lossMinus /= batchSize;

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