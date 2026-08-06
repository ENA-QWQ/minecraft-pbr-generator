package com.mc.pbr.config;

import java.util.Arrays;

public class PBRConfig {

    private GlobalConfig global = new GlobalConfig();
    private DatasetBuilderConfig datasetBuilder = new DatasetBuilderConfig();
    private TrainingConfig training = new TrainingConfig();
    private InferenceConfig inference = new InferenceConfig();

    public GlobalConfig getGlobal() {
        return global;
    }

    public void setGlobal(GlobalConfig global) {
        this.global = global;
    }

    public DatasetBuilderConfig getDatasetBuilder() {
        return datasetBuilder;
    }

    public void setDatasetBuilder(DatasetBuilderConfig datasetBuilder) {
        this.datasetBuilder = datasetBuilder;
    }

    public TrainingConfig getTraining() {
        return training;
    }

    public void setTraining(TrainingConfig training) {
        this.training = training;
    }

    public InferenceConfig getInference() {
        return inference;
    }

    public void setInference(InferenceConfig inference) {
        this.inference = inference;
    }

    public static class GlobalConfig {
        private long seed = 42;
        private int featureDim = 100;
        private int labelDim = 5;
        private int patchSize = 5;

        public long getSeed() { return seed; }
        public void setSeed(long seed) { this.seed = seed; }
        public int getFeatureDim() { return featureDim; }
        public void setFeatureDim(int featureDim) { this.featureDim = featureDim; }
        public int getLabelDim() { return labelDim; }
        public void setLabelDim(int labelDim) { this.labelDim = labelDim; }
        public int getPatchSize() { return patchSize; }
        public void setPatchSize(int patchSize) { this.patchSize = patchSize; }
    }

    public static class DatasetBuilderConfig {
        private String inputDir = "./resourcepacks";
        private String outputDir = "./dataset";
        private int maxSamples = 20000000;
        private int targetTextureSize = 128;
        private double normalOverflowRatio = 0.05;

        public String getInputDir() { return inputDir; }
        public void setInputDir(String inputDir) { this.inputDir = inputDir; }
        public String getOutputDir() { return outputDir; }
        public void setOutputDir(String outputDir) { this.outputDir = outputDir; }
        public int getMaxSamples() { return maxSamples; }
        public void setMaxSamples(int maxSamples) { this.maxSamples = maxSamples; }
        public int getTargetTextureSize() { return targetTextureSize; }
        public void setTargetTextureSize(int targetTextureSize) { this.targetTextureSize = targetTextureSize; }
        public double getNormalOverflowRatio() { return normalOverflowRatio; }
        public void setNormalOverflowRatio(double normalOverflowRatio) { this.normalOverflowRatio = normalOverflowRatio; }
    }

    public static class TrainingConfig {
        private String dataPath = "./dataset/train_data.bin";
        private String labelPath = "./dataset/train_labels.bin";
        private String modelOutput = "./height_model.ser";
        private int totalSamples = 2000000;
        private int trainSplit = 1000000;
        private int valSplit = 20000;
        private HyperparamsConfig hyperparams = new HyperparamsConfig();

        public String getDataPath() { return dataPath; }
        public void setDataPath(String dataPath) { this.dataPath = dataPath; }
        public String getLabelPath() { return labelPath; }
        public void setLabelPath(String labelPath) { this.labelPath = labelPath; }
        public String getModelOutput() { return modelOutput; }
        public void setModelOutput(String modelOutput) { this.modelOutput = modelOutput; }
        public int getTotalSamples() { return totalSamples; }
        public void setTotalSamples(int totalSamples) { this.totalSamples = totalSamples; }
        public int getTrainSplit() { return trainSplit; }
        public void setTrainSplit(int trainSplit) { this.trainSplit = trainSplit; }
        public int getValSplit() { return valSplit; }
        public void setValSplit(int valSplit) { this.valSplit = valSplit; }
        public HyperparamsConfig getHyperparams() { return hyperparams; }
        public void setHyperparams(HyperparamsConfig hyperparams) { this.hyperparams = hyperparams; }
    }

    public static class HyperparamsConfig {
        private int batchSize = 64;
        private int epochs = 50;
        private int patience = 8;
        private float learningRate = 0.01f;
        private float lrDecay = 0.9f;
        private int lrStep = 10;
        private int[] layers = new int[]{100, 32, 16, 1};

        public int getBatchSize() { return batchSize; }
        public void setBatchSize(int batchSize) { this.batchSize = batchSize; }
        public int getEpochs() { return epochs; }
        public void setEpochs(int epochs) { this.epochs = epochs; }
        public int getPatience() { return patience; }
        public void setPatience(int patience) { this.patience = patience; }
        public float getLearningRate() { return learningRate; }
        public void setLearningRate(float learningRate) { this.learningRate = learningRate; }
        public float getLrDecay() { return lrDecay; }
        public void setLrDecay(float lrDecay) { this.lrDecay = lrDecay; }
        public int getLrStep() { return lrStep; }
        public void setLrStep(int lrStep) { this.lrStep = lrStep; }
        public int[] getLayers() { return layers; }
        public void setLayers(int[] layers) { this.layers = layers; }
    }

    public static class InferenceConfig {
        private String modelPath = "./height_model.ser";
        private String outputDir = "./output";
        private float normalStrength = 6.0f;
        private boolean pixelate = false;
        private float baseSmoothness = 0.2f;
        private float baseMetallic = 0.0f;
        private HeightConfig height = new HeightConfig();
        private NormalConfig normal = new NormalConfig();

        public String getModelPath() { return modelPath; }
        public void setModelPath(String modelPath) { this.modelPath = modelPath; }
        public String getOutputDir() { return outputDir; }
        public void setOutputDir(String outputDir) { this.outputDir = outputDir; }
        public float getNormalStrength() { return normalStrength; }
        public void setNormalStrength(float normalStrength) { this.normalStrength = normalStrength; }
        public boolean isPixelate() { return pixelate; }
        public void setPixelate(boolean pixelate) { this.pixelate = pixelate; }
        public float getBaseSmoothness() { return baseSmoothness; }
        public void setBaseSmoothness(float baseSmoothness) { this.baseSmoothness = baseSmoothness; }
        public float getBaseMetallic() { return baseMetallic; }
        public void setBaseMetallic(float baseMetallic) { this.baseMetallic = baseMetallic; }
        public HeightConfig getHeight() { return height; }
        public void setHeight(HeightConfig height) { this.height = height; }
        public NormalConfig getNormal() { return normal; }
        public void setNormal(NormalConfig normal) { this.normal = normal; }
    }

    public static class HeightConfig {
        private boolean invert = true;
        private float strength = 1.2f;
        private float min = 0.2f;
        private float max = 1.0f;
        private int smoothRadius = 2;
        private float normPercentile = 2.0f;

        public boolean isInvert() { return invert; }
        public void setInvert(boolean invert) { this.invert = invert; }
        public float getStrength() { return strength; }
        public void setStrength(float strength) { this.strength = strength; }
        public float getMin() { return min; }
        public void setMin(float min) { this.min = min; }
        public float getMax() { return max; }
        public void setMax(float max) { this.max = max; }
        public int getSmoothRadius() { return smoothRadius; }
        public void setSmoothRadius(int smoothRadius) { this.smoothRadius = smoothRadius; }
        public float getNormPercentile() { return normPercentile; }
        public void setNormPercentile(float normPercentile) { this.normPercentile = normPercentile; }
    }

    public static class NormalConfig {
        private boolean invertY = false;

        public boolean isInvertY() { return invertY; }
        public void setInvertY(boolean invertY) { this.invertY = invertY; }
    }
}