package com.mc.pbr.training;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class Trainer {
    private static final int FEATURE_DIM = 100;
    private static final int LABEL_DIM = 5;
    private static final float MOMENTUM = 0.9f;

    private final String dataPath;
    private final String labelPath;
    private final int batchSize;
    private final int maxEpochs;
    private final int earlyStopPatience;
    private final float initLr;
    private final float lrDecay;
    private final int lrStepEpochs;
    private final Random rng;
    private final int totalSamples;
    private final int trainSize;
    private final int valSize;

    private float[] trainData;
    private float[] trainLabels;
    private int[] trainIndices;
    private float[] valData;
    private float[] valLabels;
    private int[] valIndices;

    public Trainer(String dataPath, String labelPath, int batchSize, int maxEpochs,
                   int earlyStopPatience, float initLr, float lrDecay, int lrStepEpochs,
                   long seed, int totalSamples, int trainSize, int valSize) {
        this.dataPath = dataPath;
        this.labelPath = labelPath;
        this.batchSize = batchSize;
        this.maxEpochs = maxEpochs;
        this.earlyStopPatience = earlyStopPatience;
        this.initLr = initLr;
        this.lrDecay = lrDecay;
        this.lrStepEpochs = lrStepEpochs;
        this.rng = new Random(seed);
        this.totalSamples = totalSamples;
        this.trainSize = trainSize;
        this.valSize = valSize;
    }

    public void prepareData() throws IOException {
        int[] allIndices = new int[totalSamples];
        for (int i = 0; i < totalSamples; i++) {
            allIndices[i] = i;
        }
        fisherYatesShuffle(allIndices, rng);

        trainIndices = new int[trainSize];
        System.arraycopy(allIndices, 0, trainIndices, 0, trainSize);

        valIndices = new int[valSize];
        System.arraycopy(allIndices, trainSize, valIndices, 0, valSize);

        int[] sortedTrainIdx = trainIndices.clone();
        Arrays.sort(sortedTrainIdx);
        int[] sortedValIdx = valIndices.clone();
        Arrays.sort(sortedValIdx);

        trainData = new float[trainSize * FEATURE_DIM];
        trainLabels = new float[trainSize * LABEL_DIM];
        valData = new float[valSize * FEATURE_DIM];
        valLabels = new float[valSize * LABEL_DIM];

        System.out.println("[INFO] Extracting training and validation subsets...");
        BinaryChunkReader.extractSamples(dataPath, labelPath, sortedTrainIdx,
                trainData, trainLabels, FEATURE_DIM, LABEL_DIM);
        BinaryChunkReader.extractSamples(dataPath, labelPath, sortedValIdx,
                valData, valLabels, FEATURE_DIM, LABEL_DIM);

        sortedTrainIdx = null;
        sortedValIdx = null;
        allIndices = null;
        System.out.println("[INFO] Data extraction completed. Memory optimized.");
    }

    private void fisherYatesShuffle(int[] array, Random rng) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }

    public void train(String heightModelPath) throws IOException {
        TinyMLP heightModel = new TinyMLP(new int[]{100, 32, 16, 1}, rng);
        System.out.println("[INFO] Architecture: Input(100) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)");
        System.out.println("[INFO] Starting Height Map model training...");
        trainModel(heightModel, trainData, trainLabels, valData, valLabels,
                2, 1, heightModelPath);
    }

    private void trainModel(TinyMLP model,
                            float[] trainData, float[] trainLabels,
                            float[] valData, float[] valLabels,
                            int labelOffset, int labelDim,
                            String savePath) throws IOException {
        int numLayers = model.getLayerSizes().length;
        int numWeightLayers = numLayers - 1;

        float[] input = new float[FEATURE_DIM];
        float[] target = new float[labelDim];

        float[][] activations = new float[numLayers][];
        activations[0] = input;
        int[] layerSizes = model.getLayerSizes();
        for (int l = 1; l < numLayers; l++) {
            activations[l] = new float[layerSizes[l]];
        }

        float[][] zs = new float[numWeightLayers][];
        for (int l = 0; l < numWeightLayers; l++) {
            zs[l] = new float[layerSizes[l + 1]];
        }

        float[][] deltas = new float[numWeightLayers][];
        for (int l = 0; l < numWeightLayers; l++) {
            deltas[l] = new float[layerSizes[l + 1]];
        }

        float[][][] gradWeights = new float[numWeightLayers][][];
        float[][] gradBiases = new float[numWeightLayers][];
        for (int l = 0; l < numWeightLayers; l++) {
            int outDim = layerSizes[l + 1];
            int inDim = layerSizes[l];
            gradWeights[l] = new float[outDim][inDim];
            gradBiases[l] = new float[outDim];
        }

        int[] localTrainIdx = new int[trainSize];
        for (int i = 0; i < trainSize; i++) localTrainIdx[i] = i;

        int[] localValIdx = new int[valSize];
        for (int i = 0; i < valSize; i++) localValIdx[i] = i;

        float[][][] bestWeights = null;
        float[][] bestBiases = null;
        float bestValLoss = Float.MAX_VALUE;
        int patienceCounter = 0;
        float lr = initLr;

        long totalStart = System.currentTimeMillis();
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();

            fisherYatesShuffle(localTrainIdx, rng);

            for (int batchStart = 0; batchStart < trainSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainSize);
                int actualBatchSize = batchEnd - batchStart;

                model.zeroGradients(gradWeights, gradBiases);

                for (int i = batchStart; i < batchEnd; i++) {
                    int idx = localTrainIdx[i];
                    System.arraycopy(trainData, idx * FEATURE_DIM, input, 0, FEATURE_DIM);
                    int labelBase = idx * LABEL_DIM + labelOffset;
                    for (int t = 0; t < labelDim; t++) {
                        target[t] = trainLabels[labelBase + t];
                    }

                    model.forward(input, activations, zs);
                    model.backward(target, activations, zs, deltas, gradWeights, gradBiases);
                }

                model.update(gradWeights, gradBiases, actualBatchSize, lr, MOMENTUM);
            }

            float trainLoss = computeLoss(model, trainData, trainLabels, localTrainIdx,
                    labelOffset, labelDim, input, target, activations, zs);

            float valLoss = computeLoss(model, valData, valLabels, localValIdx,
                    labelOffset, labelDim, input, target, activations, zs);

            long epochTime = System.currentTimeMillis() - epochStart;

            System.out.printf("[EPOCH %02d/%02d] Train MSE: %.6f | Val MSE: %.6f | Time: %d ms | LR: %.6f%n",
                    epoch, maxEpochs, trainLoss, valLoss, epochTime, lr);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                bestWeights = model.copyWeights();
                bestBiases = model.copyBiases();
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    System.out.println("[STOP] Early stopping triggered at epoch " + epoch);
                    break;
                }
            }

            if (epoch % lrStepEpochs == 0) {
                lr *= lrDecay;
            }
        }

        if (bestWeights != null) {
            model.restoreWeights(bestWeights, bestBiases);
            System.out.println("[RESTORE] Best model restored with Val MSE: " + bestValLoss);
        }

        long totalTime = System.currentTimeMillis() - totalStart;
        System.out.println("[INFO] Training completed in " + totalTime + " ms");

        saveModel(model, savePath);
    }

    private float computeLoss(TinyMLP model,
                              float[] data, float[] labels, int[] indices,
                              int labelOffset, int labelDim,
                              float[] input, float[] target,
                              float[][] activations, float[][] zs) {
        float sumSq = 0.0f;
        int n = indices.length;
        int outputLayerIdx = model.getLayerSizes().length - 1;
        float[] output = activations[outputLayerIdx];

        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            System.arraycopy(data, idx * FEATURE_DIM, input, 0, FEATURE_DIM);
            int labelBase = idx * LABEL_DIM + labelOffset;
            for (int t = 0; t < labelDim; t++) {
                target[t] = labels[labelBase + t];
            }

            model.forward(input, activations, zs);

            for (int t = 0; t < labelDim; t++) {
                float diff = output[t] - target[t];
                sumSq += diff * diff;
            }
        }
        return sumSq / n;
    }

    private void saveModel(TinyMLP model, String path) throws IOException {
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                new java.io.FileOutputStream(path))) {
            oos.writeObject(model);
        }
        System.out.println("[SAVE] Model serialized to " + path);
    }
}