package com.mc.pbr.training;

import com.mc.pbr.computing.ComputingBackend;
import com.mc.pbr.computing.BackendFactory;
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
    private final String backendType;

    private float[] trainData;
    private float[] trainLabels;
    private int[] trainIndices;
    private float[] valData;
    private float[] valLabels;
    private int[] valIndices;

    public Trainer(String dataPath, String labelPath, int batchSize, int maxEpochs,
                   int earlyStopPatience, float initLr, float lrDecay, int lrStepEpochs,
                   long seed, int totalSamples, int trainSize, int valSize, int[] layerSizes,
                   String backendType) {
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
        this.backendType = backendType;
    }

    public void prepareData() throws IOException {
        int[] allIndices = new int[totalSamples];
        for (int i = 0; i < totalSamples; i++) allIndices[i] = i;
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
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Data extraction completed.");
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
        ComputingBackend backend = BackendFactory.create(backendType, layerSizes, rng.nextLong());
        System.out.println(ANSI_BLUE + "[INFO] " + ANSI_RESET + "Backend: " + backendType.toUpperCase());

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
        System.out.println(ANSI_BLUE + "[INFO] " + ANSI_RESET + "Starting training...");

        trainModel(backend, heightModelPath);
        backend.close();
    }

    private void trainModel(ComputingBackend backend, String savePath) throws IOException {
        int labelDim = 1;
        int labelOffset = 2;

        int[] localTrainIdx = new int[trainSize];
        for (int i = 0; i < trainSize; i++) localTrainIdx[i] = i;
        int[] localValIdx = new int[valSize];
        for (int i = 0; i < valSize; i++) localValIdx[i] = i;

        float[] bestWeights = null;
        float[] bestBiases = null;
        float bestValLoss = Float.MAX_VALUE;
        int patienceCounter = 0;
        float lr = initLr;

        float[] batchInput = new float[batchSize * FEATURE_DIM];
        float[] batchLabel = new float[batchSize * labelDim];
        float[] batchOutput = new float[batchSize * labelDim];
        float[] gradOutput = new float[batchSize * labelDim];

        long totalStart = System.currentTimeMillis();
        int totalBatches = (int) Math.ceil((double) trainSize / batchSize);

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();
            fisherYatesShuffle(localTrainIdx, rng);

            int batchCount = 0;
            for (int batchStart = 0; batchStart < trainSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainSize);
                int actualBatchSize = batchEnd - batchStart;

                for (int i = batchStart; i < batchEnd; i++) {
                    int idx = localTrainIdx[i];
                    int localIdx = i - batchStart;
                    System.arraycopy(trainData, idx * FEATURE_DIM,
                            batchInput, localIdx * FEATURE_DIM, FEATURE_DIM);
                    int labelBase = idx * LABEL_DIM + labelOffset;
                    batchLabel[localIdx * labelDim] = trainLabels[labelBase];
                }

                backend.zeroGradients();
                backend.forwardBatch(batchInput, batchOutput, actualBatchSize);
                for (int i = 0; i < actualBatchSize * labelDim; i++) {
                    gradOutput[i] = batchOutput[i] - batchLabel[i];
                }
                backend.backwardBatch(batchInput, batchLabel, gradOutput, actualBatchSize);
                backend.update(null, null, actualBatchSize, lr, MOMENTUM);

                batchCount++;
                int progress = (int) (((double) batchCount / totalBatches) * 100);
                int barLength = 50;
                int filled = (int) ((progress / 100.0) * barLength);
                StringBuilder bar = new StringBuilder();
                for (int i = 0; i < barLength; i++) {
                    bar.append(i < filled ? "█" : "_");
                }
                System.out.print("\r" + ANSI_YELLOW + "[EPOCH " + String.format("%02d", epoch) +
                        "/" + String.format("%02d", maxEpochs) + "] " + ANSI_RESET +
                        "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                        progress + "% | Batch: " + batchCount + "/" + totalBatches);
            }
            System.out.println();

            float trainLoss = computeLoss(backend, trainData, trainLabels, localTrainIdx,
                    labelOffset, labelDim, ANSI_YELLOW, "[TRAIN LOSS]");
            float valLoss = computeLoss(backend, valData, valLabels, localValIdx,
                    labelOffset, labelDim, ANSI_MAGENTA, "[VALIDATE]");

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
                bestWeights = backend.getWeights();
                bestBiases = backend.getBiases();
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    System.out.println(ANSI_YELLOW + ANSI_BOLD + "[STOP] " + ANSI_RESET +
                            "Early stopping at epoch " + epoch);
                    break;
                }
            }

            if (epoch % lrStepEpochs == 0) lr *= lrDecay;
        }

        if (bestWeights != null) {
            backend.setWeights(bestWeights);
            backend.setBiases(bestBiases);
            System.out.println(ANSI_CYAN + "[RESTORE] " + ANSI_RESET +
                    "Best model restored with Val MSE: " + ANSI_GREEN + bestValLoss + ANSI_RESET);
        }

        long totalTime = System.currentTimeMillis() - totalStart;
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Training completed in " + totalTime + " ms");
        saveModel(backend, savePath);
    }

    private int calcTotalWeights() {
        int total = 0;
        for (int l = 0; l < layerSizes.length - 1; l++) {
            total += layerSizes[l] * layerSizes[l + 1];
        }
        return total;
    }

    private int calcTotalBiases() {
        int total = 0;
        for (int l = 1; l < layerSizes.length; l++) {
            total += layerSizes[l];
        }
        return total;
    }

    private float computeLoss(ComputingBackend backend,
                              float[] data, float[] labels, int[] indices,
                              int labelOffset, int labelDim,
                              String colorCode, String prefix) {
        int n = indices.length;
        int valBatchSize = Math.min(1024, n);
        float[] batchInput = new float[valBatchSize * FEATURE_DIM];
        float[] batchOutput = new float[valBatchSize * labelDim];
        float sumSq = 0.0f;
        int processed = 0;

        int barLength = 50;
        int updateInterval = Math.max(1, n / 100);
        int nextUpdate = updateInterval;

        for (int start = 0; start < n; start += valBatchSize) {
            int end = Math.min(start + valBatchSize, n);
            int actualBatch = end - start;
            for (int i = start; i < end; i++) {
                int idx = indices[i];
                System.arraycopy(data, idx * FEATURE_DIM,
                        batchInput, (i - start) * FEATURE_DIM, FEATURE_DIM);
            }
            backend.forwardBatch(batchInput, batchOutput, actualBatch);
            for (int i = 0; i < actualBatch; i++) {
                int idx = indices[start + i];
                int labelBase = idx * LABEL_DIM + labelOffset;
                float target = labels[labelBase];
                float diff = batchOutput[i * labelDim] - target;
                sumSq += diff * diff;
            }
            processed += actualBatch;
            while (processed >= nextUpdate && nextUpdate <= n) {
                int progress = (int) (((double) processed / n) * 100);
                int filled = (int) ((progress / 100.0) * barLength);
                StringBuilder bar = new StringBuilder();
                for (int i = 0; i < barLength; i++) {
                    bar.append(i < filled ? "█" : "_");
                }
                System.out.print("\r" + colorCode + prefix + " " + ANSI_RESET +
                        "[" + colorCode + bar.toString() + ANSI_RESET + "] " +
                        progress + "% | Samples: " + processed + "/" + n);
                nextUpdate += updateInterval;
                if (processed == n) {
                    System.out.println();
                    break;
                }
            }
            if (processed == n) {
                break;
            }
        }
        return sumSq / n;
    }

    private void saveModel(ComputingBackend backend, String path) throws IOException {
        float[] weights = backend.getWeights();
        float[] biases = backend.getBiases();
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                new java.io.FileOutputStream(path))) {
            oos.writeObject(new com.mc.pbr.training.ModelData(backend.getLayerSizes(), weights, biases));
        }
        System.out.println(ANSI_GREEN + "[SAVE] " + ANSI_RESET + "Model saved to " + path);
    }

}