package com.mc.pbr.training;

import com.mc.pbr.computing.ComputingBackend;
import com.mc.pbr.computing.BackendFactory;
import com.mc.pbr.computing.graph.ViTGraph;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class Trainer {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_MAGENTA = "\u001B[35m";
    private static final String ANSI_BLUE = "\u001B[34m";
    private static final String ANSI_CYAN = "\u001B[36m";

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
    private final int featureDim;
    private final int labelDim;
    private final float momentum;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private final float gradClipNorm;
    private final String modelType;
    private final int seqLen;
    private final int embedDim;
    private final int numLayers;
    private final int numHeads;
    private final int mlpDim;
    private final int inChannels;
    private final int mppNumClasses;

    private float[] trainData;
    private float[] trainLabels;
    private float[] valData;
    private float[] valLabels;

    public Trainer(String dataPath, String labelPath, int batchSize, int maxEpochs,
                   int earlyStopPatience, float initLr, float lrDecay, int lrStepEpochs,
                   long seed, int totalSamples, int trainSize, int valSize, int[] layerSizes,
                   String backendType, int featureDim, int labelDim,
                   float momentum, float beta1, float beta2, float epsilon,
                   float weightDecay, float gradClipNorm,
                   String modelType, int seqLen, int embedDim, int numLayers, int numHeads, int mlpDim, int inChannels, int mppNumClasses) {
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
        this.featureDim = featureDim;
        this.labelDim = labelDim;
        this.momentum = momentum;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.gradClipNorm = gradClipNorm;
        this.modelType = modelType;
        this.seqLen = seqLen;
        this.embedDim = embedDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.mlpDim = mlpDim;
        this.inChannels = inChannels;
        this.mppNumClasses = mppNumClasses;
    }

    public void prepareData() throws IOException {
        int[] allIndices = new int[totalSamples];
        for (int i = 0; i < totalSamples; i++) allIndices[i] = i;
        fisherYatesShuffle(allIndices);

        int[] trainIndices = new int[trainSize];
        int[] valIndices = new int[valSize];
        System.arraycopy(allIndices, 0, trainIndices, 0, trainSize);
        System.arraycopy(allIndices, trainSize, valIndices, 0, valSize);

        Arrays.sort(trainIndices);
        Arrays.sort(valIndices);

        trainData = new float[trainSize * featureDim];
        trainLabels = new float[trainSize * labelDim];
        valData = new float[valSize * featureDim];
        valLabels = new float[valSize * labelDim];

        System.out.println("[INFO] Extracting training and validation subsets...");
        BinaryChunkReader.extractSamples(dataPath, labelPath, trainIndices, trainData, trainLabels, featureDim, labelDim);
        BinaryChunkReader.extractSamples(dataPath, labelPath, valIndices, valData, valLabels, featureDim, labelDim);
        System.out.println("[INFO] Data extraction completed.");
    }

    private void fisherYatesShuffle(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = array[i];
            array[i] = array[j];
            array[j] = tmp;
        }
    }

    public void train(String heightModelPath) throws IOException {
        if ("vit".equalsIgnoreCase(modelType)) {
            trainVit(heightModelPath);
        } else {
            trainMlp(heightModelPath);
        }
    }

    private void trainMlp(String savePath) throws IOException {
        ComputingBackend backend = BackendFactory.create(backendType, layerSizes, rng.nextLong());
        System.out.println("[INFO] Backend: " + backendType.toUpperCase());
        System.out.println("[INFO] Architecture: MLP " + Arrays.toString(layerSizes));
        int heightLabelDim = 1;
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

        float[] batchInput = new float[batchSize * featureDim];
        float[] batchLabel = new float[batchSize * heightLabelDim];
        float[] batchOutput = new float[batchSize * heightLabelDim];
        float[] gradOutput = new float[batchSize * heightLabelDim];

        long totalStart = System.currentTimeMillis();
        int totalBatches = (int) Math.ceil((double) trainSize / batchSize);

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();
            fisherYatesShuffle(localTrainIdx);

            int batchCount = 0;
            for (int batchStart = 0; batchStart < trainSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainSize);
                int actualBatchSize = batchEnd - batchStart;

                for (int i = batchStart; i < batchEnd; i++) {
                    int idx = localTrainIdx[i];
                    int localIdx = i - batchStart;
                    System.arraycopy(trainData, idx * featureDim, batchInput, localIdx * featureDim, featureDim);
                    int labelBase = idx * labelDim + labelOffset;
                    batchLabel[localIdx * heightLabelDim] = trainLabels[labelBase];
                }

                backend.zeroGradients();
                backend.forwardBatch(batchInput, batchOutput, actualBatchSize);
                for (int i = 0; i < actualBatchSize * heightLabelDim; i++) {
                    gradOutput[i] = batchOutput[i] - batchLabel[i];
                }
                backend.backwardBatch(batchInput, batchLabel, gradOutput, actualBatchSize);
                backend.update(null, null, actualBatchSize, lr, momentum);

                batchCount++;
                printTrainingProgress(epoch, batchCount, totalBatches, epochStart);
            }
            System.out.println();

            float trainLoss = computeLossMlp(backend, trainData, trainLabels, localTrainIdx, labelOffset, heightLabelDim, true);
            float valLoss = computeLossMlp(backend, valData, valLabels, localValIdx, labelOffset, heightLabelDim, false);

            long epochTime = System.currentTimeMillis() - epochStart;
            printEpochSummary(epoch, trainLoss, valLoss, epochTime, lr);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                bestWeights = backend.getWeights();
                bestBiases = backend.getBiases();
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    System.out.println("[STOP] Early stopping at epoch " + epoch);
                    break;
                }
            }

            if (epoch % lrStepEpochs == 0) lr *= lrDecay;
        }

        if (bestWeights != null) {
            backend.setWeights(bestWeights);
            backend.setBiases(bestBiases);
            System.out.println("[RESTORE] Best model restored with Val MSE: " + bestValLoss);
        }

        long totalTime = System.currentTimeMillis() - totalStart;
        System.out.println("[INFO] Training completed in " + totalTime + " ms");
        saveModel(backend, savePath);
        backend.close();
    }

    private void trainVit(String savePath) throws IOException {
        ViTGraph vit = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, rng.nextLong(), mppNumClasses);
        System.out.println("[INFO] Architecture: ViT (embed=" + embedDim + ", layers=" + numLayers + ", heads=" + numHeads + ", mlp=" + mlpDim + ", seq=" + seqLen + ")");
        int totalBatches = (int) Math.ceil((double) trainSize / batchSize);
        float lr = initLr;

        float[] bestWeights = null;
        float[] bestBiases = null;
        float bestValLoss = Float.MAX_VALUE;
        int patienceCounter = 0;
        int step = 0;

        float[] batchInput = new float[batchSize * featureDim];
        float[] batchLabel = new float[batchSize * labelDim];
        float[] batchOutput = new float[batchSize * labelDim];
        float[] gradOutput = new float[batchSize * labelDim];

        long totalStart = System.currentTimeMillis();

        int[] localTrainIdx = new int[trainSize];
        for (int i = 0; i < trainSize; i++) localTrainIdx[i] = i;
        int[] localValIdx = new int[valSize];
        for (int i = 0; i < valSize; i++) localValIdx[i] = i;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();
            fisherYatesShuffle(localTrainIdx);

            int batchCount = 0;
            for (int batchStart = 0; batchStart < trainSize; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, trainSize);
                int actualBatchSize = batchEnd - batchStart;

                for (int i = batchStart; i < batchEnd; i++) {
                    int idx = localTrainIdx[i];
                    int localIdx = i - batchStart;
                    System.arraycopy(trainData, idx * featureDim, batchInput, localIdx * featureDim, featureDim);
                    System.arraycopy(trainLabels, idx * labelDim, batchLabel, localIdx * labelDim, labelDim);
                }

                vit.zeroGradients();
                vit.forward(batchInput, batchOutput, actualBatchSize);
                for (int i = 0; i < actualBatchSize * labelDim; i++) {
                    gradOutput[i] = batchOutput[i] - batchLabel[i];
                }
                vit.backward(batchInput, batchLabel, gradOutput, actualBatchSize);
                vit.clipGradients(gradClipNorm);
                vit.adamwUpdate(actualBatchSize, lr, beta1, beta2, epsilon, weightDecay, step);
                step++;

                batchCount++;
                printTrainingProgress(epoch, batchCount, totalBatches, epochStart);
            }
            System.out.println();

            float trainLoss = computeLossVit(vit, trainData, trainLabels, localTrainIdx, true);
            float valLoss = computeLossVit(vit, valData, valLabels, localValIdx, false);

            long epochTime = System.currentTimeMillis() - epochStart;
            printEpochSummary(epoch, trainLoss, valLoss, epochTime, lr);

            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
                bestWeights = vit.getWeights();
                bestBiases = vit.getBiases();
            } else {
                patienceCounter++;
                if (patienceCounter >= earlyStopPatience) {
                    System.out.println("[STOP] Early stopping at epoch " + epoch);
                    break;
                }
            }

            if (epoch % lrStepEpochs == 0) lr *= lrDecay;
        }

        if (bestWeights != null) {
            vit.setWeights(bestWeights);
            vit.setBiases(bestBiases);
            System.out.println("[RESTORE] Best model restored with Val MSE: " + bestValLoss);
        }

        long totalTime = System.currentTimeMillis() - totalStart;
        System.out.println("[INFO] Training completed in " + totalTime + " ms");
        saveModelVit(vit, savePath);
        vit.close();
    }

    private float computeLossMlp(ComputingBackend backend, float[] data, float[] labels, int[] indices,
                                 int labelOffset, int heightLabelDim, boolean isTrain) {
        int n = indices.length;
        int valBatchSize = Math.min(1024, n);
        float[] batchInput = new float[valBatchSize * featureDim];
        float[] batchOutput = new float[valBatchSize * heightLabelDim];
        float sumSq = 0.0f;
        int processed = 0;
        int barLength = 50;
        int updateInterval = Math.max(1, n / 100);
        String color = isTrain ? ANSI_YELLOW : ANSI_MAGENTA;
        String prefix = isTrain ? "TRAIN LOSS" : "VALIDATE";
        long startTime = System.currentTimeMillis();

        printLossProgress(prefix, color, 0, 0, n, 0, startTime);

        for (int start = 0; start < n; start += valBatchSize) {
            int end = Math.min(start + valBatchSize, n);
            int actualBatch = end - start;
            for (int i = start; i < end; i++) {
                int idx = indices[i];
                System.arraycopy(data, idx * featureDim, batchInput, (i - start) * featureDim, featureDim);
            }
            backend.forwardBatch(batchInput, batchOutput, actualBatch);
            for (int i = 0; i < actualBatch; i++) {
                int idx = indices[start + i];
                int labelBase = idx * labelDim + labelOffset;
                float target = labels[labelBase];
                float diff = batchOutput[i * heightLabelDim] - target;
                sumSq += diff * diff;
            }
            processed += actualBatch;
            if (processed % updateInterval == 0 || processed >= n) {
                int progress = (int) (((double) processed / n) * 100);
                long elapsed = System.currentTimeMillis() - startTime;
                printLossProgress(prefix, color, progress, processed, n, elapsed, startTime);
                if (processed >= n) System.out.println();
            }
        }
        return sumSq / n;
    }

    private float computeLossVit(ViTGraph vit, float[] data, float[] labels, int[] indices, boolean isTrain) {
        int n = indices.length;
        int valBatchSize = Math.min(1024, n);
        float[] batchInput = new float[valBatchSize * featureDim];
        float[] batchOutput = new float[valBatchSize * labelDim];
        float sumSq = 0.0f;
        int processed = 0;
        int barLength = 50;
        int updateInterval = Math.max(1, n / 100);
        String color = isTrain ? ANSI_YELLOW : ANSI_MAGENTA;
        String prefix = isTrain ? "TRAIN LOSS" : "VALIDATE";
        long startTime = System.currentTimeMillis();

        printLossProgress(prefix, color, 0, 0, n, 0, startTime);

        for (int start = 0; start < n; start += valBatchSize) {
            int end = Math.min(start + valBatchSize, n);
            int actualBatch = end - start;
            for (int i = start; i < end; i++) {
                int idx = indices[i];
                System.arraycopy(data, idx * featureDim, batchInput, (i - start) * featureDim, featureDim);
            }
            vit.forward(batchInput, batchOutput, actualBatch);
            for (int i = 0; i < actualBatch; i++) {
                int idx = indices[start + i];
                int labelBase = idx * labelDim;
                for (int p = 0; p < labelDim; p++) {
                    float diff = batchOutput[i * labelDim + p] - labels[labelBase + p];
                    sumSq += diff * diff;
                }
            }
            processed += actualBatch;
            if (processed % updateInterval == 0 || processed >= n) {
                int progress = (int) (((double) processed / n) * 100);
                long elapsed = System.currentTimeMillis() - startTime;
                printLossProgress(prefix, color, progress, processed, n, elapsed, startTime);
                if (processed >= n) System.out.println();
            }
        }
        return sumSq / (n * labelDim);
    }

    private void printLossProgress(String prefix, String color, int progress, int processed, int total, long elapsed, long startTime) {
        int barLength = 50;
        int filled = (int) ((progress / 100.0) * barLength);
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < barLength; i++) {
            bar.append(i < filled ? "█" : "_");
        }
        String timeStr = formatDuration(elapsed);
        System.out.print("\r" + ANSI_YELLOW + "[" + prefix + "] " + ANSI_RESET +
                "[" + color + bar.toString() + ANSI_RESET + "] " +
                progress + "% | Samples: " + processed + "/" + total +
                " | " + ANSI_CYAN + "Elapsed: " + timeStr + ANSI_RESET);
        System.out.flush();
    }

    private void printTrainingProgress(int epoch, int batch, int total, long epochStart) {
        int progress = (int) (((double) batch / total) * 100);
        int barLength = 50;
        int filled = (int) ((progress / 100.0) * barLength);
        StringBuilder bar = new StringBuilder();
        for (int i = 0; i < barLength; i++) {
            bar.append(i < filled ? "█" : "_");
        }
        long elapsed = System.currentTimeMillis() - epochStart;
        long eta = 0;
        if (progress > 0) {
            eta = (elapsed * (100 - progress)) / progress;
        }
        String elapsedStr = formatDuration(elapsed);
        String etaStr = (progress == 100) ? "0s" : formatDuration(eta);
        System.out.print("\r" + ANSI_YELLOW + "[EPOCH " + String.format("%02d", epoch) + "/" + String.format("%02d", maxEpochs) + "] " + ANSI_RESET +
                "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                progress + "% | Batch: " + batch + "/" + total +
                " | " + ANSI_CYAN + "Elapsed: " + elapsedStr + " | ETA: " + etaStr + ANSI_RESET);
        System.out.flush();
    }

    private void printEpochSummary(int epoch, float trainLoss, float valLoss, long epochTime, float lr) {
        String valStr = String.format("%.6f", valLoss);
        System.out.printf(ANSI_BLUE + "[EPOCH %02d/%02d] " + ANSI_RESET +
                        ANSI_YELLOW + "Train MSE: %.6f " + ANSI_RESET + "| " +
                        ANSI_GREEN + "Val MSE: %s " + ANSI_RESET + "| " +
                        ANSI_CYAN + "Time: %d ms " + ANSI_RESET + "| " +
                        ANSI_MAGENTA + "LR: %.6f%n" + ANSI_RESET,
                epoch, maxEpochs, trainLoss, valStr, epochTime, lr);
    }

    private String formatDuration(long millis) {
        long seconds = millis / 1000;
        long minutes = seconds / 60;
        long hours = minutes / 60;
        seconds %= 60;
        minutes %= 60;
        if (hours > 0) {
            return String.format("%dh%02dm%02ds", hours, minutes, seconds);
        } else if (minutes > 0) {
            return String.format("%dm%02ds", minutes, seconds);
        } else {
            return String.format("%ds", seconds);
        }
    }

    private void saveModel(ComputingBackend backend, String path) throws IOException {
        float[] weights = backend.getWeights();
        float[] biases = backend.getBiases();
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(path))) {
            oos.writeObject(new ModelData("mlp", backend.getLayerSizes(), weights, biases, 0, 0, 0, 0, 0, 0, 0));
        }
        System.out.println("[SAVE] Model saved to " + path);
    }

    private void saveModelVit(ViTGraph vit, String path) throws IOException {
        float[] weights = vit.getWeights();
        float[] biases = vit.getBiases();
        int[] layerSizes = new int[]{featureDim, embedDim, numLayers, numHeads, mlpDim};
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(new java.io.FileOutputStream(path))) {
            oos.writeObject(new ModelData("vit", layerSizes, weights, biases, seqLen, embedDim, numLayers, numHeads, mlpDim, inChannels, mppNumClasses));
        }
        System.out.println("[SAVE] Model saved to " + path);
    }
}