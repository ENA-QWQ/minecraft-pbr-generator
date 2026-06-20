package com.mc.pbr.inference;

public class HeightToNormalConverter {

    public void convert(float[] heightMap, int w, int h, float strength, boolean invertY, int[] normalPixels) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;

                float hLeft = heightMap[y * w + ((x - 1 + w) % w)];
                float hRight = heightMap[y * w + ((x + 1) % w)];
                float hUp = heightMap[((y - 1 + h) % h) * w + x];
                float hDown = heightMap[((y + 1) % h) * w + x];

                float dzdx = (hRight - hLeft) * strength;
                float dzdy = (hDown - hUp) * strength;

                if (invertY) {
                    dzdy = -dzdy;
                }

                float nx = -dzdx;
                float ny = -dzdy;
                float nz = 1.0f;

                float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                nx /= len;
                ny /= len;
                nz /= len;

                int outR = clamp((nx * 0.5f + 0.5f) * 255.0f);
                int outG = clamp((ny * 0.5f + 0.5f) * 255.0f);
                int outA = clamp(heightMap[idx] * 191.0f + 64.0f);
                int outB = 255;

                normalPixels[idx] = (outA << 24) | (outR << 16) | (outG << 8) | outB;
            }
        }
    }

    private int clamp(float val) {
        return Math.max(0, Math.min(255, (int) val));
    }
}