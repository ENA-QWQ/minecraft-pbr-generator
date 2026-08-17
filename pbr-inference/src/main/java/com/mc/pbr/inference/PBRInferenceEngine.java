package com.mc.pbr.inference;

import com.mc.pbr.computing.ComputingBackend;
import com.mc.pbr.computing.BackendFactory;
import com.mc.pbr.computing.graph.ViTGraph;
import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;

public class PBRInferenceEngine {
    private final ComputingBackend backend;
    private final ViTGraph vitGraph;
    private final String modelType;
    private final float strength;
    private final boolean pixelate;
    private final float baseSmoothness;
    private final float baseMetallic;
    private final boolean invertHeight;
    private final boolean invertNormalY;
    private final float heightStrength;
    private final float heightMin;
    private final float heightMax;
    private final int heightSmoothRadius;
    private final float normPercentile;
    private final int featureDim;
    private final int labelDim;
    private final int patchRadius;
    private final int inferenceBatchSize;
    private final int seqLen;
    private final int embedDim;
    private final int numLayers;
    private final int numHeads;
    private final int mlpDim;
    private final int inChannels;
    private final HeightToNormalConverter normalConverter = new HeightToNormalConverter();
    private final NearestNeighborScaler scaler = new NearestNeighborScaler();

    public PBRInferenceEngine(ModelLoader modelLoader, float strength, boolean pixelate,
                              float baseSmoothness, float baseMetallic,
                              boolean invertHeight, boolean invertNormalY,
                              float heightStrength, float heightMin, float heightMax,
                              int heightSmoothRadius, float normPercentile,
                              String backendType, int patchRadius, int inferenceBatchSize,
                              String modelType, int seqLen, int embedDim, int numLayers, int numHeads, int mlpDim, int inChannels) {
        this.modelType = modelType;
        this.seqLen = seqLen;
        this.embedDim = embedDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.mlpDim = mlpDim;
        this.inChannels = inChannels;
        this.patchRadius = patchRadius;
        this.inferenceBatchSize = inferenceBatchSize;
        this.strength = strength;
        this.pixelate = pixelate;
        this.baseSmoothness = baseSmoothness;
        this.baseMetallic = baseMetallic;
        this.invertHeight = invertHeight;
        this.invertNormalY = invertNormalY;
        this.heightStrength = heightStrength;
        this.heightMin = heightMin;
        this.heightMax = heightMax;
        this.heightSmoothRadius = heightSmoothRadius;
        this.normPercentile = normPercentile;

        if ("vit".equalsIgnoreCase(modelType)) {
            this.vitGraph = new ViTGraph(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels,
                    modelLoader.getWeights(), modelLoader.getBiases(), 512);
            this.backend = null;
            this.featureDim = seqLen * inChannels;
            this.labelDim = seqLen;
        } else {
            this.backend = BackendFactory.createFromWeights(
                    backendType,
                    modelLoader.getLayerSizes(),
                    modelLoader.getWeights(),
                    modelLoader.getBiases()
            );
            this.vitGraph = null;
            this.featureDim = backend.getFeatureDim();
            this.labelDim = backend.getLabelDim();
        }
    }

    public void process(String inputPath, String outputDir) throws Exception {
        File inputFile = new File(inputPath);
        if (!inputFile.exists()) throw new IOException("Input image not found: " + inputPath);

        BufferedImage origImg = ImageIO.read(inputFile);
        if (origImg == null) throw new IOException("Unsupported image format: " + inputPath);

        if (origImg.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage converted = new BufferedImage(origImg.getWidth(), origImg.getHeight(),
                    BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = converted.createGraphics();
            g2d.drawImage(origImg, 0, 0, null);
            g2d.dispose();
            origImg = converted;
        }

        int origW = origImg.getWidth();
        int origH = origImg.getHeight();

        int targetW = origW, targetH = origH;
        int scale = 1;
        if (origW < 128 || origH < 128) {
            while (origW * scale < 128 || origH * scale < 128) scale *= 2;
            targetW = origW * scale;
            targetH = origH * scale;
        }

        BufferedImage workImg = origImg;
        if (scale > 1) {
            workImg = new BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = workImg.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            g2d.drawImage(origImg, 0, 0, targetW, targetH, null);
            g2d.dispose();
        }

        int w = workImg.getWidth();
        int h = workImg.getHeight();
        int totalPixels = w * h;
        int[] pixels = ((DataBufferInt) workImg.getRaster().getDataBuffer()).getData();

        float[] heightMap = new float[totalPixels];
        int[] normalPixels = new int[totalPixels];
        int[] matPixels = new int[totalPixels];

        if ("vit".equalsIgnoreCase(modelType)) {
            processVit(workImg, pixels, w, h, totalPixels, heightMap, normalPixels, matPixels);
        } else {
            processMlp(workImg, pixels, w, h, totalPixels, heightMap, normalPixels, matPixels);
        }

        File outDirFile = new File(outputDir);
        if (!outDirFile.exists()) outDirFile.mkdirs();

        saveImage(normalPixels, w, h, new File(outDirFile, "texture_n.png"));
        saveImage(matPixels, w, h, new File(outDirFile, "texture_s.png"));
        System.out.println("[SAVE] Results saved to: " + outDirFile.getAbsolutePath());

        if (backend != null) backend.close();
        if (vitGraph != null) vitGraph.close();
    }

    private void processMlp(BufferedImage workImg, int[] pixels, int w, int h, int totalPixels,
                            float[] heightMap, int[] normalPixels, int[] matPixels) throws Exception {
        System.out.print("[INFO] Pass 1: Feature extraction & MLP inference... ");
        long pass1Start = System.currentTimeMillis();

        int batchSize = Math.min(inferenceBatchSize, totalPixels);
        float[] allFeatures = new float[totalPixels * featureDim];
        int idx = 0;
        int updateInterval = Math.max(1, totalPixels / 50);
        int processed = 0;
        int patchSize = 2 * patchRadius + 1;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int dy = -patchRadius; dy <= patchRadius; dy++) {
                    for (int dx = -patchRadius; dx <= patchRadius; dx++) {
                        int sx = (x + dx + w) % w;
                        int sy = (y + dy + h) % h;
                        int pixel = pixels[sy * w + sx];
                        float r = ((pixel >> 16) & 0xFF) / 255.0f;
                        float g = ((pixel >> 8) & 0xFF) / 255.0f;
                        float b = (pixel & 0xFF) / 255.0f;
                        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                        allFeatures[idx++] = r;
                        allFeatures[idx++] = g;
                        allFeatures[idx++] = b;
                        allFeatures[idx++] = gray;
                    }
                }
                processed++;
                if (processed % updateInterval == 0 || processed == totalPixels) {
                    int progress = (int) (((double) processed / totalPixels) * 100);
                    int barLength = 30;
                    int filled = (int) ((progress / 100.0) * barLength);
                    StringBuilder bar = new StringBuilder();
                    for (int i = 0; i < barLength; i++) {
                        if (i < filled) bar.append("█");
                        else bar.append("_");
                    }
                    System.out.print("\r[INFO] [ " + bar.toString() + " ] " +
                            progress + "% | Pixel: " + processed + "/" + totalPixels);
                }
            }
        }
        System.out.println();

        float[] allHeights = new float[totalPixels];
        backend.forwardBatch(allFeatures, allHeights, totalPixels);

        for (int i = 0; i < totalPixels; i++) {
            heightMap[i] = allHeights[i];
            int pixel = pixels[i];
            float gray = 0.299f * (((pixel >> 16) & 0xFF) / 255.0f) +
                    0.587f * (((pixel >> 8) & 0xFF) / 255.0f) +
                    0.114f * ((pixel & 0xFF) / 255.0f);
            float smoothness = baseSmoothness + gray * 0.3f;
            float metallic = (((pixel >> 0) & 0xFF) / 255.0f) > 0.5f ? baseMetallic : 0.0f;
            int outS = clamp(smoothness * 255.0f);
            int outM = clamp(metallic * 255.0f);
            matPixels[i] = 0xFF000000 | (outS << 16) | (outM << 8);
        }

        System.out.println("[INFO] Pass 1 completed in " + (System.currentTimeMillis() - pass1Start) + " ms");
        postProcess(heightMap, normalPixels, matPixels, pixels, w, h, totalPixels);
    }

    private void processVit(BufferedImage workImg, int[] pixels, int w, int h, int totalPixels,
                            float[] heightMap, int[] normalPixels, int[] matPixels) throws Exception {
        System.out.print("[INFO] ViT inference: extracting patches... ");
        int patchSize = (int) Math.sqrt((double) (w * h) / seqLen);
        int actualPatchSize = patchSize;
        while (actualPatchSize * actualPatchSize * seqLen > w * h) actualPatchSize--;
        int channels = 4;
        float[] allFeatures = new float[seqLen * actualPatchSize * actualPatchSize * channels];
        int idx = 0;
        for (int py = 0; py < h; py += actualPatchSize) {
            for (int px = 0; px < w; px += actualPatchSize) {
                if (idx / (actualPatchSize * actualPatchSize * channels) >= seqLen) break;
                for (int y = py; y < py + actualPatchSize && y < h; y++) {
                    for (int x = px; x < px + actualPatchSize && x < w; x++) {
                        int pixel = pixels[y * w + x];
                        float r = ((pixel >> 16) & 0xFF) / 255.0f;
                        float g = ((pixel >> 8) & 0xFF) / 255.0f;
                        float b = (pixel & 0xFF) / 255.0f;
                        float a = ((pixel >> 24) & 0xFF) / 255.0f;
                        allFeatures[idx++] = r;
                        allFeatures[idx++] = g;
                        allFeatures[idx++] = b;
                        allFeatures[idx++] = a;
                    }
                }
            }
        }
        System.out.println("Done.");

        float[] patchHeights = new float[seqLen];
        vitGraph.forward(allFeatures, patchHeights, 1);

        for (int py = 0; py < h; py += actualPatchSize) {
            for (int px = 0; px < w; px += actualPatchSize) {
                int patchIdx = (py / actualPatchSize) * (w / actualPatchSize) + (px / actualPatchSize);
                if (patchIdx >= seqLen) break;
                float hVal = patchHeights[patchIdx];
                for (int y = py; y < py + actualPatchSize && y < h; y++) {
                    for (int x = px; x < px + actualPatchSize && x < w; x++) {
                        int i = y * w + x;
                        heightMap[i] = hVal;
                        int pixel = pixels[i];
                        float gray = 0.299f * (((pixel >> 16) & 0xFF) / 255.0f) +
                                0.587f * (((pixel >> 8) & 0xFF) / 255.0f) +
                                0.114f * ((pixel & 0xFF) / 255.0f);
                        float smoothness = baseSmoothness + gray * 0.3f;
                        float metallic = (((pixel >> 0) & 0xFF) / 255.0f) > 0.5f ? baseMetallic : 0.0f;
                        int outS = clamp(smoothness * 255.0f);
                        int outM = clamp(metallic * 255.0f);
                        matPixels[i] = 0xFF000000 | (outS << 16) | (outM << 8);
                    }
                }
            }
        }

        postProcess(heightMap, normalPixels, matPixels, pixels, w, h, totalPixels);
    }

    private void postProcess(float[] heightMap, int[] normalPixels, int[] matPixels, int[] pixels,
                             int w, int h, int totalPixels) {
        System.out.print("[INFO] Post-processing height map... ");
        applyPercentileNormalization(heightMap, totalPixels);

        if (invertHeight) {
            for (int i = 0; i < totalPixels; i++) heightMap[i] = 1.0f - heightMap[i];
        }

        if (heightSmoothRadius > 0) {
            applyBoxBlur(heightMap, w, h, heightSmoothRadius);
        }

        float stretchedMin = Float.MAX_VALUE, stretchedMax = -Float.MAX_VALUE;
        for (int i = 0; i < totalPixels; i++) {
            float stretchedH = 0.5f + (heightMap[i] - 0.5f) * heightStrength;
            heightMap[i] = stretchedH;
            if (stretchedH < stretchedMin) stretchedMin = stretchedH;
            if (stretchedH > stretchedMax) stretchedMax = stretchedH;
        }

        float stretchedRange = stretchedMax - stretchedMin;
        if (stretchedRange < 1e-5f) stretchedRange = 1e-5f;
        float targetRange = heightMax - heightMin;

        for (int i = 0; i < totalPixels; i++) {
            heightMap[i] = heightMin + (heightMap[i] - stretchedMin) / stretchedRange * targetRange;
        }
        System.out.println("Done.");

        System.out.print("[INFO] Converting height map to normal map... ");
        normalConverter.convert(heightMap, w, h, strength, invertNormalY, normalPixels);
        System.out.println("Done.");
    }

    private void applyPercentileNormalization(float[] arr, int n) {
        float[] sorted = arr.clone();
        java.util.Arrays.sort(sorted);
        int lowIdx = (int) (n * (normPercentile / 100.0f));
        int highIdx = (int) (n * (1.0f - normPercentile / 100.0f));
        if (lowIdx >= n) lowIdx = n - 1;
        if (highIdx >= n) highIdx = n - 1;
        float minVal = sorted[lowIdx];
        float maxVal = sorted[highIdx];
        float range = maxVal - minVal;
        if (range < 1e-5f) range = 1e-5f;
        for (int i = 0; i < n; i++) {
            arr[i] = (arr[i] - minVal) / range;
            arr[i] = Math.max(0.0f, Math.min(1.0f, arr[i]));
        }
    }

    private void applyBoxBlur(float[] src, int w, int h, int radius) {
        float[] dst = new float[src.length];
        int kernelSize = (2 * radius + 1) * (2 * radius + 1);
        float invKernel = 1.0f / kernelSize;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int sx = (x + dx + w) % w;
                        int sy = (y + dy + h) % h;
                        sum += src[sy * w + sx];
                    }
                }
                dst[y * w + x] = sum * invKernel;
            }
        }
        System.arraycopy(dst, 0, src, 0, src.length);
    }

    private void saveImage(int[] pixels, int w, int h, File file) throws IOException {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        int[] imgPixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
        System.arraycopy(pixels, 0, imgPixels, 0, pixels.length);
        ImageIO.write(img, "png", file);
    }

    private int clamp(float val) {
        return Math.max(0, Math.min(255, (int) val));
    }
}