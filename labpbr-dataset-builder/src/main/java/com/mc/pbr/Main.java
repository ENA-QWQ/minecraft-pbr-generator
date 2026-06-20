package com.mc.pbr;

import java.io.*;
import java.util.*;
import java.util.logging.*;

public class Main {

    public static void main(String[] args) {
        String inputDir = "./resourcepacks";
        String outputDir = "./dataset";
        int maxSamples = 20_000_000;
        long seed = System.currentTimeMillis();

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-inputDir": inputDir = args[++i]; break;
                case "-outputDir": outputDir = args[++i]; break;
                case "-maxSamples": maxSamples = Integer.parseInt(args[++i]); break;
                case "-seed": seed = Long.parseLong(args[++i]); break;
                default:
                    System.out.println("[ERROR] Usage: java -jar app.jar -inputDir <path> -outputDir <path> [-maxSamples N] [-seed N]");
                    return;
            }
        }

        File root = new File(inputDir);
        if (!root.isDirectory()) {
            System.out.println("[ERROR] Input directory does not exist: " + root.getAbsolutePath());
            return;
        }
        new File(outputDir).mkdirs();

        System.out.println("[INFO] Input: " + root.getAbsolutePath());
        System.out.println("[INFO] Output: " + new File(outputDir).getAbsolutePath());
        System.out.println("[INFO] Max samples: " + maxSamples);
        System.out.println("[INFO] Random seed: " + seed);
        System.out.println("[INFO] Strategy: All texture groups contribute evenly, stream-write as we go");

        System.out.println("[INFO] Step 1/4: Validating and cleaning data...");
        LabPBRValidator validator = new LabPBRValidator();
        List<LabPBRValidator.TextureTriple> triples;
        try {
            triples = validator.validateAndCollect(root);
        } catch (IOException e) {
            System.out.println("[ERROR] Failed to scan resource packs");
            return;
        }
        int packCount = triples.size();
        System.out.println("[INFO] Valid texture groups: " + packCount);
        if (packCount == 0) {
            System.out.println("[ERROR] No valid texture groups found. Exiting.");
            return;
        }

        final int FEATURE_DIM = 100;
        final int LABEL_DIM = 5;

        int[] quotas = new int[packCount];
        int baseQuota = maxSamples / packCount;
        int remainder = maxSamples % packCount;
        for (int i = 0; i < packCount; i++) {
            quotas[i] = baseQuota + (i < remainder ? 1 : 0);
        }
        System.out.printf("[INFO] Quota per group: average %d, first %d get +1 extra%n", baseQuota, remainder);

        DatasetSerializer serializer = new DatasetSerializer();
        try {
            serializer.open(
                    new File(outputDir, "train_data.bin"),
                    new File(outputDir, "train_labels.bin"),
                    FEATURE_DIM,
                    LABEL_DIM
            );
        } catch (IOException e) {
            System.out.println("[ERROR] Failed to create output files");
            return;
        }

        System.out.println("[INFO] Step 2+3/4: Extracting per texture, random sampling and writing to disk...");
        LabPBRDataExtractor extractor = new LabPBRDataExtractor();
        DataAugmenter augmenter = new DataAugmenter();
        Random rng = new Random(seed);
        int totalWritten = 0;
        int processedTextures = 0;

        try {
            for (int packIdx = 0; packIdx < packCount; packIdx++) {
                LabPBRValidator.TextureTriple triple = triples.get(packIdx);
                LabPBRDataExtractor.ExtractionResult result = extractor.extract(triple);
                if (result == null) {
                    processedTextures++;
                    continue;
                }

                List<float[]> features = result.features;
                List<float[]> labels = result.labels;
                int pixelCount = features.size();

                int quota = quotas[packIdx];
                for (int s = 0; s < quota; s++) {
                    int pixelIdx = rng.nextInt(pixelCount);
                    int augType = rng.nextInt(8);

                    float[] feat = features.get(pixelIdx);
                    float[] label = labels.get(pixelIdx);

                    float[] augFeat = augmenter.applyAugmentation(feat, augType);
                    float[] augLabel = augmenter.applyLabelAugmentation(label, augType);

                    serializer.writeSample(augFeat, augLabel);
                    totalWritten++;
                }

                processedTextures++;
                if (processedTextures % 20 == 0) {
                    System.out.printf("[INFO] Progress: %d/%d textures, samples written: %d%n",
                            processedTextures, packCount, totalWritten);
                }
            }
        } catch (IOException e) {
            System.out.println("[ERROR] Error while writing samples");
        } finally {
            try {
                serializer.close();
            } catch (IOException e) {
                System.out.println("[ERROR] Failed to close serializer");
            }
        }

        System.out.printf("[INFO] Total samples actually written: %d%n", totalWritten);
        System.out.println("[INFO] Files saved to: " + outputDir);
        System.out.println("[INFO] Processing complete.");
    }
}