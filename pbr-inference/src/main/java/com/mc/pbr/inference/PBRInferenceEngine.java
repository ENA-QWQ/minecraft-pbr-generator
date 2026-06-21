package com.mc.pbr.inference;
import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class PBRInferenceEngine {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_CYAN = "\u001B[36m";

    private final ModelLoader heightModel;
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
    private final float[] featureFloat = new float[100];
    private final double[] featureDouble = new double[100];
    private final HeightToNormalConverter normalConverter = new HeightToNormalConverter();
    private final NearestNeighborScaler scaler = new NearestNeighborScaler();

    public PBRInferenceEngine(ModelLoader heightModel, float strength, boolean pixelate,
                              float baseSmoothness, float baseMetallic,
                              boolean invertHeight, boolean invertNormalY,
                              float heightStrength, float heightMin, float heightMax,
                              int heightSmoothRadius, float normPercentile) {
        this.heightModel = heightModel;
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
    }

    public void process(String inputPath, String outputDir) throws Exception {
        File inputFile = new File(inputPath);
        if (!inputFile.exists()) throw new IOException("Input image not found: " + inputPath);

        BufferedImage origImg = ImageIO.read(inputFile);
        if (origImg == null) throw new IOException("Unsupported image format: " + inputPath);

        if (origImg.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage convertedImg = new BufferedImage(origImg.getWidth(), origImg.getHeight(), BufferedImage.TYPE_INT_ARGB);
            Graphics2D g2d = convertedImg.createGraphics();
            g2d.drawImage(origImg, 0, 0, null);
            g2d.dispose();
            origImg = convertedImg;
        }

        int origW = origImg.getWidth();
        int origH = origImg.getHeight();

        int targetW = origW, targetH = origH;
        int scale = 1;
        if (origW < 128 || origH < 128) {
            while (origW * scale < 128 || origH * scale < 128) {
                scale *= 2;
            }
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

        System.out.print(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pass 1: Feature extraction & MLP inference... ");
        long pass1Start = System.currentTimeMillis();
        int updateInterval = Math.max(1, totalPixels / 50);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = 0;
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        int sx = (x + dx + w) % w;
                        int sy = (y + dy + h) % h;
                        int pixel = pixels[sy * w + sx];
                        float r = ((pixel >> 16) & 0xFF) / 255.0f;
                        float g = ((pixel >> 8) & 0xFF) / 255.0f;
                        float b = (pixel & 0xFF) / 255.0f;
                        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                        featureFloat[idx++] = r;
                        featureFloat[idx++] = g;
                        featureFloat[idx++] = b;
                        featureFloat[idx++] = gray;
                    }
                }

                double[] geomOut = heightModel.predict(featureFloat, featureDouble);
                heightMap[y * w + x] = (float) geomOut[0];

                float gray = 0.299f * (((pixels[y * w + x] >> 16) & 0xFF) / 255.0f) +
                        0.587f * (((pixels[y * w + x] >> 8) & 0xFF) / 255.0f) +
                        0.114f * ((pixels[y * w + x] & 0xFF) / 255.0f);

                float smoothness = baseSmoothness + gray * 0.3f;
                float metallic = (((pixels[y * w + x] >> 0) & 0xFF) / 255.0f) > 0.5f ? baseMetallic : 0.0f;

                int outS = clamp(smoothness * 255.0f);
                int outM = clamp(metallic * 255.0f);
                matPixels[y * w + x] = 0xFF000000 | (outS << 16) | (outM << 8);

                int currentPixel = y * w + x + 1;
                if (currentPixel % updateInterval == 0 || currentPixel == totalPixels) {
                    int progress = (int) (((double) currentPixel / totalPixels) * 100);
                    int barLength = 30;
                    int filled = (int) ((progress / 100.0) * barLength);
                    StringBuilder bar = new StringBuilder();
                    for (int i = 0; i < barLength; i++) {
                        if (i < filled) bar.append("█");
                        else bar.append("_");
                    }
                    System.out.print("\r" + ANSI_CYAN + "[INFO] " + ANSI_RESET +
                            "[ " + ANSI_GREEN + bar.toString() + ANSI_RESET + " ] " +
                            progress + "% | Pixel: " + currentPixel + "/" + totalPixels);
                }
            }
        }
        System.out.println();
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pass 1 completed in " + (System.currentTimeMillis() - pass1Start) + " ms");

        System.out.print(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pass 2: Normalization & Post-processing... ");
        applyPercentileNormalization(heightMap, totalPixels);

        if (invertHeight) {
            for (int i = 0; i < totalPixels; i++) {
                heightMap[i] = 1.0f - heightMap[i];
            }
        }

        if (heightSmoothRadius > 0) {
            applyBoxBlur(heightMap, w, h, heightSmoothRadius);
        }

        float stretchedMin = Float.MAX_VALUE;
        float stretchedMax = -Float.MAX_VALUE;
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
            float finalH = heightMin + (heightMap[i] - stretchedMin) / stretchedRange * targetRange;
            heightMap[i] = finalH;
        }
        System.out.println("Done.");

        System.out.print(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Converting height map to normal map... ");
        normalConverter.convert(heightMap, w, h, strength, invertNormalY, normalPixels);
        System.out.println("Done.");

        if (scale > 1 && pixelate) {
            System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Applying hard-edge pixelation...");
            float[] smallHeight = new float[origW * origH];
            float[] tempHeight = new float[totalPixels];
            scaler.scale(heightMap, w, h, origW, origH, smallHeight);
            scaler.scale(smallHeight, origW, origH, w, h, tempHeight);

            for (int i = 0; i < totalPixels; i++) {
                int pixel = normalPixels[i];
                int outA = clamp(tempHeight[i] * 191.0f + 64.0f);
                normalPixels[i] = (pixel & 0x00FFFFFF) | (outA << 24);
            }

            int[] smallMat = new int[origW * origH];
            int[] tempMat = new int[totalPixels];
            scaler.scale(matPixels, w, h, origW, origH, smallMat);
            scaler.scale(smallMat, origW, origH, w, h, tempMat);
            System.arraycopy(tempMat, 0, matPixels, 0, totalPixels);
        }

        File outDirFile = new File(outputDir);
        if (!outDirFile.exists()) outDirFile.mkdirs();

        saveImage(normalPixels, w, h, new File(outDirFile, "texture_n.png"));
        saveImage(matPixels, w, h, new File(outDirFile, "texture_s.png"));
        System.out.println(ANSI_GREEN + "[SAVE] " + ANSI_RESET + "Results saved to: " + outDirFile.getAbsolutePath());
    }

    private void applyPercentileNormalization(float[] arr, int n) {
        float[] sorted = arr.clone();
        Arrays.sort(sorted);
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