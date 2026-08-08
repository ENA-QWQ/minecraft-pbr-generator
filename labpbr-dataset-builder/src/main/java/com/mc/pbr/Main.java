package com.mc.pbr;

import com.mc.pbr.config.ConfigLoader;
import com.mc.pbr.config.PBRConfig;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class Main {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_MAGENTA = "\u001B[35m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_BOLD = "\u001B[1m";

    public static void main(String[] args) {
        PBRConfig config = ConfigLoader.load(args);

        System.out.println(ANSI_MAGENTA + ANSI_BOLD);
        System.out.println("███╗   ███╗ ██████╗██████╗  ██████╗ ");
        System.out.println("████╗ ████║██╔════╝██╔══██╗██╔════╝ ");
        System.out.println("██╔████╔██║██║     ██████╔╝██║  ███╗");
        System.out.println("██║╚██╔╝██║██║     ██╔═══╝ ██║   ██║");
        System.out.println("██║ ╚═╝ ██║╚██████╗██║     ╚██████╔╝");
        System.out.println("╚═╝     ╚═╝ ╚═════╝╚═╝      ╚═════╝ ");
        System.out.println(ANSI_RESET);

        PBRConfig.DatasetBuilderConfig db = config.getDatasetBuilder();
        PBRConfig.GlobalConfig gb = config.getGlobal();

        File root = new File(db.getInputDir());
        if (!root.isDirectory()) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Input directory does not exist: " + root.getAbsolutePath());
            System.exit(1);
        }
        new File(db.getOutputDir()).mkdirs();

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pipeline execution started.");
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Configuration:" + ANSI_RESET);
        System.out.printf("  %-12s: %s\n", "Input", root.getAbsolutePath());
        System.out.printf("  %-12s: %s\n", "Output", new File(db.getOutputDir()).getAbsolutePath());
        System.out.printf("  %-12s: %d\n", "MaxSamples", db.getMaxSamples());
        System.out.printf("  %-12s: %d\n", "Seed", gb.getSeed());
        System.out.printf("  %-12s: %d\n", "PatchRadius", gb.getPatchSize());
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Strategy: All texture groups contribute evenly, stream-write as we go");

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Step 1/4: Validating and cleaning data...");
        LabPBRValidator validator = new LabPBRValidator();
        List<LabPBRValidator.TextureTriple> triples;
        try {
            triples = validator.validateAndCollect(root);
        } catch (IOException e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Failed to scan resource packs");
            e.printStackTrace();
            System.exit(1);
            return;
        }
        int packCount = triples.size();
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Valid texture groups: " + ANSI_GREEN + packCount + ANSI_RESET);
        if (packCount == 0) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "No valid texture groups found. Exiting.");
            System.exit(1);
            return;
        }

        int patchRadius = gb.getPatchSize();
        int featureDim = (2 * patchRadius + 1) * (2 * patchRadius + 1) * 4;
        gb.setFeatureDim(featureDim);
        final int FEATURE_DIM = gb.getFeatureDim();
        final int LABEL_DIM = gb.getLabelDim();

        int[] quotas = new int[packCount];
        int baseQuota = db.getMaxSamples() / packCount;
        int remainder = db.getMaxSamples() % packCount;
        for (int i = 0; i < packCount; i++) {
            quotas[i] = baseQuota + (i < remainder ? 1 : 0);
        }
        System.out.printf(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Quota per group: average " + ANSI_GREEN + "%d" + ANSI_RESET +
                        ", first " + ANSI_GREEN + "%d" + ANSI_RESET + " get +1 extra%n",
                baseQuota, remainder);

        DatasetSerializer serializer = new DatasetSerializer();
        try {
            serializer.open(
                    new File(db.getOutputDir(), "train_data.bin"),
                    new File(db.getOutputDir(), "train_labels.bin"),
                    FEATURE_DIM,
                    LABEL_DIM
            );
        } catch (IOException e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Failed to create output files");
            e.printStackTrace();
            System.exit(1);
            return;
        }

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Step 2+3/4: Extracting per texture, random sampling and writing to disk...");
        LabPBRDataExtractor extractor = new LabPBRDataExtractor(patchRadius);
        DataAugmenter augmenter = new DataAugmenter(patchRadius);
        Random rng = new Random(gb.getSeed());
        int totalWritten = 0;
        int processedTextures = 0;
        int barLength = 50;

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
                int progress = (int) (((double) processedTextures / packCount) * 100);
                int filled = (int) ((progress / 100.0) * barLength);
                StringBuilder bar = new StringBuilder();
                for (int k = 0; k < barLength; k++) {
                    if (k < filled) bar.append("█");
                    else bar.append("_");
                }
                System.out.print("\r" + ANSI_CYAN + "[BUILD] " + ANSI_RESET +
                        "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                        progress + "% | Textures: " + processedTextures + "/" + packCount +
                        " | Samples: " + totalWritten);
            }
            System.out.println();
        } catch (IOException e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Error while writing samples");
            e.printStackTrace();
        } finally {
            try {
                serializer.close();
            } catch (IOException e) {
                System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Failed to close serializer");
                e.printStackTrace();
            }
        }

        System.out.printf(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Total samples actually written: " + ANSI_GREEN + "%d%n" + ANSI_RESET, totalWritten);
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Files saved to: " + db.getOutputDir());
        System.out.println(ANSI_GREEN + ANSI_BOLD + "[INFO] " + ANSI_RESET + "Pipeline execution completed successfully.");
    }
}