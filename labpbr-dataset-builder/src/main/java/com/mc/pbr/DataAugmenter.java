package com.mc.pbr;

import java.util.*;

public class DataAugmenter {
    private final int patchRadius;
    private final int patchSize;

    public DataAugmenter(int patchRadius) {
        this.patchRadius = patchRadius;
        this.patchSize = 2 * patchRadius + 1;
    }

    public float[] applyAugmentation(float[] flatFeature, int type) {
        if (type == 0 || type == 7) {
            return flatFeature.clone();
        }
        float[][][] patch = reshapeFeature(flatFeature);
        float[][][] transformed;
        switch (type) {
            case 1:
                transformed = rotate90Clockwise(patch);
                break;
            case 2:
                transformed = rotate180(patch);
                break;
            case 3:
                transformed = rotate270Clockwise(patch);
                break;
            case 4:
                transformed = flipHorizontal(patch);
                break;
            case 5:
                transformed = flipVertical(patch);
                break;
            case 6:
                transformed = flipVertical(flipHorizontal(patch));
                break;
            default:
                return flatFeature.clone();
        }
        return flattenFeature(transformed);
    }

    public float[] applyLabelAugmentation(float[] label, int type) {
        float[] l = label.clone();
        float nx = label[0];
        float ny = label[1];
        switch (type) {
            case 1:
                l[0] = ny;
                l[1] = -nx;
                break;
            case 2:
                l[0] = -nx;
                l[1] = -ny;
                break;
            case 3:
                l[0] = -ny;
                l[1] = nx;
                break;
            case 4:
                l[0] = -nx;
                break;
            case 5:
                l[1] = -ny;
                break;
            case 6:
                l[0] = -nx;
                l[1] = -ny;
                break;
        }
        return l;
    }

    private float[][][] reshapeFeature(float[] flat) {
        float[][][] patch = new float[patchSize][patchSize][4];
        int idx = 0;
        for (int y = 0; y < patchSize; y++)
            for (int x = 0; x < patchSize; x++)
                for (int c = 0; c < 4; c++)
                    patch[y][x][c] = flat[idx++];
        return patch;
    }

    private float[] flattenFeature(float[][][] patch) {
        float[] flat = new float[patchSize * patchSize * 4];
        int idx = 0;
        for (int y = 0; y < patchSize; y++)
            for (int x = 0; x < patchSize; x++)
                for (int c = 0; c < 4; c++)
                    flat[idx++] = patch[y][x][c];
        return flat;
    }

    private float[][][] rotate90Clockwise(float[][][] patch) {
        float[][][] res = new float[patchSize][patchSize][4];
        for (int y = 0; y < patchSize; y++)
            for (int x = 0; x < patchSize; x++)
                res[x][patchSize - 1 - y] = patch[y][x];
        return res;
    }

    private float[][][] rotate180(float[][][] patch) {
        return rotate90Clockwise(rotate90Clockwise(patch));
    }

    private float[][][] rotate270Clockwise(float[][][] patch) {
        return rotate90Clockwise(rotate90Clockwise(rotate90Clockwise(patch)));
    }

    private float[][][] flipHorizontal(float[][][] patch) {
        float[][][] res = new float[patchSize][patchSize][4];
        for (int y = 0; y < patchSize; y++)
            for (int x = 0; x < patchSize; x++)
                res[y][patchSize - 1 - x] = patch[y][x];
        return res;
    }

    private float[][][] flipVertical(float[][][] patch) {
        float[][][] res = new float[patchSize][patchSize][4];
        for (int y = 0; y < patchSize; y++)
            res[patchSize - 1 - y] = patch[y];
        return res;
    }
}