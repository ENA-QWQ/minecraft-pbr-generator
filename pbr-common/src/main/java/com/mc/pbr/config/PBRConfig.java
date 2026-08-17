package com.mc.pbr.config;

public class PBRConfig {
    private GlobalConfig global = new GlobalConfig();
    private DatasetBuilderConfig datasetBuilder = new DatasetBuilderConfig();
    private TrainingConfig training = new TrainingConfig();
    private InferenceConfig inference = new InferenceConfig();

    public GlobalConfig getGlobal() { return global; }
    public void setGlobal(GlobalConfig global) { this.global = global; }
    public DatasetBuilderConfig getDatasetBuilder() { return datasetBuilder; }
    public void setDatasetBuilder(DatasetBuilderConfig datasetBuilder) { this.datasetBuilder = datasetBuilder; }
    public TrainingConfig getTraining() { return training; }
    public void setTraining(TrainingConfig training) { this.training = training; }
    public InferenceConfig getInference() { return inference; }
    public void setInference(InferenceConfig inference) { this.inference = inference; }

    public static class GlobalConfig {
        private long seed = 42;
        private int featureDim = 100;
        private int labelDim = 5;
        private int patchSize = 5;
        private String backend = "cpu";
        private GPUConfig gpu = new GPUConfig();
        private String modelType = "mlp";
        private int seqLen = 16;
        private int embedDim = 128;
        private int numLayers = 4;
        private int numHeads = 8;
        private int mlpDim = 256;
        private int inChannels = 4;

        public String getModelType() { return modelType; }
        public void setModelType(String modelType) { this.modelType = modelType; }
        public int getSeqLen() { return seqLen; }
        public void setSeqLen(int seqLen) { this.seqLen = seqLen; }
        public int getEmbedDim() { return embedDim; }
        public void setEmbedDim(int embedDim) { this.embedDim = embedDim; }
        public int getNumLayers() { return numLayers; }
        public void setNumLayers(int numLayers) { this.numLayers = numLayers; }
        public int getNumHeads() { return numHeads; }
        public void setNumHeads(int numHeads) { this.numHeads = numHeads; }
        public int getMlpDim() { return mlpDim; }
        public void setMlpDim(int mlpDim) { this.mlpDim = mlpDim; }
        public int getInChannels() { return inChannels; }
        public void setInChannels(int inChannels) { this.inChannels = inChannels; }
        public long getSeed() { return seed; }
        public void setSeed(long seed) { this.seed = seed; }
        public int getFeatureDim() { return featureDim; }
        public void setFeatureDim(int featureDim) { this.featureDim = featureDim; }
        public int getLabelDim() { return labelDim; }
        public void setLabelDim(int labelDim) { this.labelDim = labelDim; }
        public int getPatchSize() { return patchSize; }
        public void setPatchSize(int patchSize) { this.patchSize = patchSize; }
        public String getBackend() { return backend; }
        public void setBackend(String backend) { this.backend = backend; }
        public GPUConfig getGpu() { return gpu; }
        public void setGpu(GPUConfig gpu) { this.gpu = gpu; }
        private String datasetType = "mlp";
        private int vitTrainBatchSize = 8;
        private float vitLearningRate = 0.001f;
        private float vitWeightDecay = 0.1f;
        private float vitGradClip = 1.0f;

        public String getDatasetType() { return datasetType; }
        public void setDatasetType(String datasetType) { this.datasetType = datasetType; }
        public int getVitTrainBatchSize() { return vitTrainBatchSize; }
        public void setVitTrainBatchSize(int vitTrainBatchSize) { this.vitTrainBatchSize = vitTrainBatchSize; }
        public float getVitLearningRate() { return vitLearningRate; }
        public void setVitLearningRate(float vitLearningRate) { this.vitLearningRate = vitLearningRate; }
        public float getVitWeightDecay() { return vitWeightDecay; }
        public void setVitWeightDecay(float vitWeightDecay) { this.vitWeightDecay = vitWeightDecay; }
        public float getVitGradClip() { return vitGradClip; }
        public void setVitGradClip(float vitGradClip) { this.vitGradClip = vitGradClip; }
    }

    public static class GPUConfig {
        private int maxBatchSize = 65536;
        private int workgroupSize = 256;
        private int mppNumClasses = 512;
        private float weightInitStd = 0.02f;

        public int getMaxBatchSize() { return maxBatchSize; }
        public void setMaxBatchSize(int maxBatchSize) { this.maxBatchSize = maxBatchSize; }
        public int getWorkgroupSize() { return workgroupSize; }
        public void setWorkgroupSize(int workgroupSize) { this.workgroupSize = workgroupSize; }
        public int getMppNumClasses() { return mppNumClasses; }
        public void setMppNumClasses(int mppNumClasses) { this.mppNumClasses = mppNumClasses; }
        public float getWeightInitStd() { return weightInitStd; }
        public void setWeightInitStd(float weightInitStd) { this.weightInitStd = weightInitStd; }
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
        private float momentum = 0.9f;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-8f;
        private float weightDecay = 0.1f;
        private float gradientClipNorm = 1.0f;

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
        public float getMomentum() { return momentum; }
        public void setMomentum(float momentum) { this.momentum = momentum; }
        public float getBeta1() { return beta1; }
        public void setBeta1(float beta1) { this.beta1 = beta1; }
        public float getBeta2() { return beta2; }
        public void setBeta2(float beta2) { this.beta2 = beta2; }
        public float getEpsilon() { return epsilon; }
        public void setEpsilon(float epsilon) { this.epsilon = epsilon; }
        public float getWeightDecay() { return weightDecay; }
        public void setWeightDecay(float weightDecay) { this.weightDecay = weightDecay; }
        public float getGradientClipNorm() { return gradientClipNorm; }
        public void setGradientClipNorm(float gradientClipNorm) { this.gradientClipNorm = gradientClipNorm; }
    }

    public static class InferenceConfig {
        private String modelPath = "./height_model.ser";
        private String outputDir = "./output";
        private float normalStrength = 6.0f;
        private boolean pixelate = false;
        private float baseSmoothness = 0.2f;
        private float baseMetallic = 0.0f;
        private int inferenceBatchSize = 1024;
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
        public int getInferenceBatchSize() { return inferenceBatchSize; }
        public void setInferenceBatchSize(int inferenceBatchSize) { this.inferenceBatchSize = inferenceBatchSize; }
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