package com.mc.pbr.inference;

public class NearestNeighborScaler {

    public void scale(float[] src, int srcW, int srcH, int dstW, int dstH, float[] dst) {
        for (int y = 0; y < dstH; y++) {
            int srcY = y * srcH / dstH;
            for (int x = 0; x < dstW; x++) {
                int srcX = x * srcW / dstW;
                dst[y * dstW + x] = src[srcY * srcW + srcX];
            }
        }
    }

    public void scale(int[] src, int srcW, int srcH, int dstW, int dstH, int[] dst) {
        for (int y = 0; y < dstH; y++) {
            int srcY = y * srcH / dstH;
            for (int x = 0; x < dstW; x++) {
                int srcX = x * srcW / dstW;
                dst[y * dstW + x] = src[srcY * srcW + srcX];
            }
        }
    }
}