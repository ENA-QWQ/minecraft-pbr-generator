package com.mc.pbr.training;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class Trainer {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_BLUE = "\u001B[34m";
    private static final String ANSI_MAGENTA = "\u001B[35m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_BOLD = "\u001B[1m";
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
    private final int[] layerSizes;

    private float[] trainData;
    private float[] trainLabels;
    private int[] trainIndices;
    private float[] valData;
    private float[] valLabels;
    private int[] valIndices;

    public Trainer(String dataPath, String labelPath, int batchSize, int maxEpochs,
                   int earlyStopPatience, float initLr, float lrDecay, int lrStepEpochs,
                   long seed, int totalSamples, int trainSize, int valSize, int[] layerSizes) {
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
        this.layerSizes = layerSizes;
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

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Extracting training and validation subsets...");
        BinaryChunkReader.extractSamples(dataPath, labelPath, sortedTrainIdx,
                trainData, trainLabels, FEATURE_DIM, LABEL_DIM);
        BinaryChunkReader.extractSamples(dataPath, labelPath, sortedValIdx,
                valData, valLabels, FEATURE_DIM, LABEL_DIM);

        sortedTrainIdx = null;
        sortedValIdx = null;
        allIndices = null;
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Data extraction completed. Memory optimized.");
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
        TinyMLP heightModel = new TinyMLP(layerSizes, rng);

        StringBuilder arch = new StringBuilder();
        arch.append("Input(").append(layerSizes[0]).append(")");
        for (int i = 1; i < layerSizes.length; i++) {
            if (i == layerSizes.length - 1) {
                arch.append(" -> Dense(").append(layerSizes[i]).append(", Linear)");
            } else {
                arch.append(" -> Dense(").append(layerSizes[i]).append(", ReLU)");
            }
        }

        System.out.println(ANSI_BLUE + "[INFO] " + ANSI_RESET + "Architecture: " + arch.toString());
        System.out.println(ANSI_BLUE + "[INFO] " + ANSI_RESET + "Starting Height Map model training...");

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
        int totalBatches = (int) Math.ceil((double) trainSize / batchSize);

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();

            fisherYatesShuffle(localTrainIdx, rng);

            int batchCount = 0;
            for (int batchStart = 0; batchStart < trainSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainSize);
                int actualBatchSize = batchEnd - batchStart;
                batchCount++;

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

                int progress = (int) (((double) batchCount / totalBatches) * 100);
                int barLength = 50;
                int filled = (int) ((progress / 100.0) * barLength);
                StringBuilder bar = new StringBuilder();
                for (int i = 0; i < barLength; i++) {
                    if (i < filled) bar.append("█");
                    else bar.append("_");
                }
                System.out.print("\r" + ANSI_YELLOW + "[EPOCH " + String.format("%02d", epoch) + "/" + String.format("%02d", maxEpochs) + "] " +
                        ANSI_RESET + "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                        progress + "% | Batch: " + batchCount + "/" + totalBatches);
            }
            System.out.println();

            float trainLoss = computeLoss(model, trainData, trainLabels, localTrainIdx,
                    labelOffset, labelDim, input, target, activations, zs, ANSI_YELLOW, "[TRAIN LOSS]");

            float valLoss = computeLoss(model, valData, valLabels, localValIdx,
                    labelOffset, labelDim, input, target, activations, zs, ANSI_MAGENTA, "[VALIDATE]");

            long epochTime = System.currentTimeMillis() - epochStart;

            String valLossStr = String.format("%.6f", valLoss);
            if (valLoss < bestValLoss) {
                valLossStr = ANSI_GREEN + ANSI_BOLD + valLossStr + " (Best)" + ANSI_RESET;
            }

            System.out.printf(ANSI_BLUE + "[EPOCH %02d/%02d] " + ANSI_RESET +
                            ANSI_YELLOW + "Train MSE: %.6f " + ANSI_RESET + "| " +
                            ANSI_GREEN + "Val MSE: %s " + ANSI_RESET + "| " +
                            ANSI_CYAN + "Time: %d ms " + ANSI_RESET + "| " +
                            ANSI_MAGENTA + "LR: %.6f%n" + ANSI_RESET,
                    epoch, maxEpochs, trainLoss, valLossStr, epochTime, lr);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                bestWeights = model.copyWeights();
                bestBiases = model.copyBiases();
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    System.out.println(ANSI_YELLOW + ANSI_BOLD + "[STOP] " + ANSI_RESET + "Early stopping triggered at epoch " + epoch);
                    break;
                }
            }

            if (epoch % lrStepEpochs == 0) {
                lr *= lrDecay;
            }
        }

        if (bestWeights != null) {
            model.restoreWeights(bestWeights, bestBiases);
            System.out.println(ANSI_CYAN + "[RESTORE] " + ANSI_RESET + "Best model restored with Val MSE: " + ANSI_GREEN + bestValLoss + ANSI_RESET);
        }

        long totalTime = System.currentTimeMillis() - totalStart;
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Training completed in " + totalTime + " ms");

        saveModel(model, savePath);
    }

    private float computeLoss(TinyMLP model,
                              float[] data, float[] labels, int[] indices,
                              int labelOffset, int labelDim,
                              float[] input, float[] target,
                              float[][] activations, float[][] zs,
                              String colorCode, String prefix) {
        float sumSq = 0.0f;
        int n = indices.length;
        int outputLayerIdx = model.getLayerSizes().length - 1;
        float[] output = activations[outputLayerIdx];

        int barLength = 50;
        int updateInterval = Math.max(1, n / 100);
        int processed = 0;

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

            processed++;
            if (processed % updateInterval == 0 || processed == n) {
                int progress = (int) (((double) processed / n) * 100);
                int filled = (int) ((progress / 100.0) * barLength);
                StringBuilder bar = new StringBuilder();
                for (int k = 0; k < barLength; k++) {
                    if (k < filled) bar.append("█");
                    else bar.append("_");
                }
                System.out.print("\r" + ANSI_YELLOW + prefix + " " + ANSI_RESET +
                        "[" + colorCode + bar.toString() + ANSI_RESET + "] " +
                        progress + "% | Samples: " + processed + "/" + n);
                if (processed == n) {
                    System.out.println();
                }
            }
        }
        return sumSq / n;
    }

    private void saveModel(TinyMLP model, String path) throws IOException {
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                new java.io.FileOutputStream(path))) {
            oos.writeObject(model);
        }
        System.out.println(ANSI_GREEN + "[SAVE] " + ANSI_RESET + "Model serialized to " + path);
    }
}