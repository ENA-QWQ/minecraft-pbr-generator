package com.mc.pbr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LabPBRDataExtractor {
    private final int patchRadius;
    private final int targetSize;
    private final int seqLen;
    private final String modelType;

    public LabPBRDataExtractor(int patchRadius, int targetSize, int seqLen, String modelType) {
        this.patchRadius = patchRadius;
        this.targetSize = targetSize;
        this.seqLen = seqLen;
        this.modelType = modelType;
    }

    public List<Sample> extract(LabPBRValidator.TextureTriple triple) {
        try {
            BufferedImage baseImg = resizeToTarget(ImageIO.read(triple.base));
            BufferedImage normalImg = resizeToTarget(ImageIO.read(triple.normal));
            BufferedImage specImg = resizeToTarget(ImageIO.read(triple.specular));

            int w = baseImg.getWidth();
            int h = baseImg.getHeight();
            if (w != targetSize || h != targetSize) {
                System.out.println("[WARNING] Resized image size abnormal: " + triple.base.getName());
                return null;
            }

            int[] basePixels = baseImg.getRGB(0, 0, w, h, null, 0, w);
            int[] normalPixels = normalImg.getRGB(0, 0, w, h, null, 0, w);
            int[] specPixels = specImg.getRGB(0, 0, w, h, null, 0, w);

            float[][] baseNorm = pixelArrayToNormalized(basePixels);
            float[][] normalNorm = pixelArrayToNormalized(normalPixels);
            float[][] specNorm = pixelArrayToNormalized(specPixels);

            List<Sample> samples = new ArrayList<>();

            if ("vit".equalsIgnoreCase(modelType)) {
                int patchSize = (int) Math.sqrt((double) (w * h) / seqLen);
                if (patchSize * patchSize * seqLen != w * h) {
                    while (patchSize * patchSize * seqLen > w * h) patchSize--;
                }
                int channels = 4;
                float[] feat = new float[seqLen * patchSize * patchSize * channels];
                float[] label = new float[seqLen];
                int patchIdx = 0;
                for (int py = 0; py < h && patchIdx < seqLen; py += patchSize) {
                    for (int px = 0; px < w && patchIdx < seqLen; px += patchSize) {
                        int idx = 0;
                        float sumHeight = 0.0f;
                        int count = 0;
                        for (int y = py; y < py + patchSize && y < h; y++) {
                            for (int x = px; x < px + patchSize && x < w; x++) {
                                int pi = y * w + x;
                                feat[patchIdx * patchSize * patchSize * channels + idx++] = baseNorm[0][pi];
                                feat[patchIdx * patchSize * patchSize * channels + idx++] = baseNorm[1][pi];
                                feat[patchIdx * patchSize * patchSize * channels + idx++] = baseNorm[2][pi];
                                feat[patchIdx * patchSize * patchSize * channels + idx++] = baseNorm[3][pi];
                                float height = normalNorm[3][pi] * 2.0f - 1.0f;
                                sumHeight += height;
                                count++;
                            }
                        }
                        label[patchIdx] = sumHeight / count;
                        patchIdx++;
                    }
                }
                samples.add(new Sample(feat, label));
            } else {
                int patchSize = 2 * patchRadius + 1;
                int channels = 4;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        float[] feat = extractPatch(baseNorm, x, y, w, h);
                        float[] label = extractLabel(normalNorm, specNorm, x, y);
                        samples.add(new Sample(feat, label));
                    }
                }
            }

            return samples;
        } catch (IOException e) {
            System.out.println("[ERROR] Failed to extract texture triple: " + triple.base);
            return null;
        }
    }

    private BufferedImage resizeToTarget(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w == targetSize && h == targetSize) {
            return src;
        }
        BufferedImage resized = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(src, 0, 0, targetSize, targetSize, null);
        g.dispose();
        return resized;
    }

    private float[][] pixelArrayToNormalized(int[] pixels) {
        int len = pixels.length;
        float[] r = new float[len];
        float[] g = new float[len];
        float[] b = new float[len];
        float[] a = new float[len];
        for (int i = 0; i < len; i++) {
            int p = pixels[i];
            r[i] = ((p >> 16) & 0xFF) / 255.0f;
            g[i] = ((p >> 8) & 0xFF) / 255.0f;
            b[i] = (p & 0xFF) / 255.0f;
            a[i] = ((p >> 24) & 0xFF) / 255.0f;
        }
        return new float[][]{r, g, b, a};
    }

    private float[] extractPatch(float[][] baseNorm, int cx, int cy, int w, int h) {
        int patchSize = 2 * patchRadius + 1;
        float[] feat = new float[patchSize * patchSize * 4];
        int idx = 0;
        for (int dy = -patchRadius; dy <= patchRadius; dy++) {
            for (int dx = -patchRadius; dx <= patchRadius; dx++) {
                int nx = (cx + dx + w) % w;
                int ny = (cy + dy + h) % h;
                int pi = ny * w + nx;
                feat[idx++] = baseNorm[0][pi];
                feat[idx++] = baseNorm[1][pi];
                feat[idx++] = baseNorm[2][pi];
                feat[idx++] = baseNorm[3][pi];
            }
        }
        return feat;
    }

    private float[] extractLabel(float[][] normalNorm, float[][] specNorm, int x, int y) {
        int idx = y * targetSize + x;
        float height = normalNorm[3][idx] * 2.0f - 1.0f;
        return new float[]{height};
    }

    public static class Sample {
        public final float[] features;
        public final float[] labels;

        public Sample(float[] features, float[] labels) {
            this.features = features;
            this.labels = labels;
        }
    }
}