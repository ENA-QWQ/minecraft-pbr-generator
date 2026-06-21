package com.mc.pbr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LabPBRDataExtractor {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_BOLD = "\u001B[1m";

    private static final int TARGET_SIZE = 128;
    private static final int PATCH_RADIUS = 2;

    public ExtractionResult extract(LabPBRValidator.TextureTriple triple) {
        try {
            BufferedImage baseImg = resizeToTarget(ImageIO.read(triple.base));
            BufferedImage normalImg = resizeToTarget(ImageIO.read(triple.normal));
            BufferedImage specImg = resizeToTarget(ImageIO.read(triple.specular));

            int w = baseImg.getWidth();
            int h = baseImg.getHeight();
            if (w != TARGET_SIZE || h != TARGET_SIZE) {
                System.out.println(ANSI_YELLOW + "[WARNING] " + ANSI_RESET + "Resized image size abnormal: " + triple.base.getName());
                return null;
            }

            int[] basePixels = baseImg.getRGB(0, 0, w, h, null, 0, w);
            int[] normalPixels = normalImg.getRGB(0, 0, w, h, null, 0, w);
            int[] specPixels = specImg.getRGB(0, 0, w, h, null, 0, w);

            float[][] baseNorm = pixelArrayToNormalized(basePixels);
            float[][] normalNorm = pixelArrayToNormalized(normalPixels);
            float[][] specNorm = pixelArrayToNormalized(specPixels);

            List<float[]> features = new ArrayList<>();
            List<float[]> labels = new ArrayList<>();

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float[] feat = extractPatch(baseNorm, x, y, w, h);
                    float[] label = extractLabel(normalNorm, specNorm, x, y, w, h);
                    features.add(feat);
                    labels.add(label);
                }
            }

            return new ExtractionResult(features, labels);
        } catch (IOException e) {
            System.out.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Failed to extract texture triple: " + triple.base);
            return null;
        }
    }

    private BufferedImage resizeToTarget(BufferedImage src) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w == TARGET_SIZE && h == TARGET_SIZE) {
            return src;
        }
        BufferedImage resized = new BufferedImage(TARGET_SIZE, TARGET_SIZE, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(src, 0, 0, TARGET_SIZE, TARGET_SIZE, null);
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
        float[] feat = new float[25 * 4];
        int idx = 0;
        float[] r = baseNorm[0];
        float[] g = baseNorm[1];
        float[] b = baseNorm[2];
        for (int dy = -PATCH_RADIUS; dy <= PATCH_RADIUS; dy++) {
            for (int dx = -PATCH_RADIUS; dx <= PATCH_RADIUS; dx++) {
                int nx = (cx + dx + w) % w;
                int ny = (cy + dy + h) % h;
                int pi = ny * w + nx;
                feat[idx++] = r[pi];
                feat[idx++] = g[pi];
                feat[idx++] = b[pi];
                feat[idx++] = 0.299f * r[pi] + 0.587f * g[pi] + 0.114f * b[pi];
            }
        }
        return feat;
    }

    private float[] extractLabel(float[][] normalNorm, float[][] specNorm, int x, int y, int w, int h) {
        int idx = y * w + x;
        float nx = normalNorm[0][idx] * 2.0f - 1.0f;
        float ny = normalNorm[1][idx] * 2.0f - 1.0f;
        float height = normalNorm[3][idx] * 2.0f - 1.0f;
        float smoothness = specNorm[0][idx];
        float metallic = specNorm[1][idx];
        return new float[]{nx, ny, height, smoothness, metallic};
    }

    public static class ExtractionResult {
        public final List<float[]> features;
        public final List<float[]> labels;

        public ExtractionResult(List<float[]> features, List<float[]> labels) {
            this.features = features;
            this.labels = labels;
        }
    }
}