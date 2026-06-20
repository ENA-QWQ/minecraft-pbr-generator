package com.mc.pbr.training;

import java.util.HashMap;
import java.util.Map;

public class TrainMain {
    public static void main(String[] args) {
        Map<String, String> params = parseArgs(args);
        if (params.containsKey("help")) {
            printHelp();
            return;
        }

        String dataPath = params.getOrDefault("data", "train_data.bin");
        String labelPath = params.getOrDefault("labels", "train_labels.bin");
        String outputPath = params.getOrDefault("output", "height_model.ser");
        int batchSize = Integer.parseInt(params.getOrDefault("batch-size", "64"));
        int maxEpochs = Integer.parseInt(params.getOrDefault("epochs", "50"));
        int patience = Integer.parseInt(params.getOrDefault("patience", "8"));
        float initLr = Float.parseFloat(params.getOrDefault("lr", "0.01"));
        float lrDecay = Float.parseFloat(params.getOrDefault("lr-decay", "0.9"));
        int lrStep = Integer.parseInt(params.getOrDefault("lr-step", "10"));
        long seed = Long.parseLong(params.getOrDefault("seed", "42"));
        int totalSamples = Integer.parseInt(params.getOrDefault("total-samples", "2000000"));
        int trainSize = Integer.parseInt(params.getOrDefault("train-size", "1000000"));
        int valSize = Integer.parseInt(params.getOrDefault("val-size", "20000"));

        if (trainSize + valSize > totalSamples) {
            System.err.println("[ERROR] train-size + val-size cannot exceed total-samples");
            System.exit(1);
        }

        System.out.println("[INFO] Pipeline execution started.");
        System.out.println("[INFO] Data: " + dataPath + " | " + labelPath);
        System.out.println("[INFO] Output: " + outputPath);
        System.out.println("[INFO] Hyperparameters -> BatchSize: " + batchSize +
                ", Epochs: " + maxEpochs + ", Patience: " + patience +
                ", LR: " + initLr + ", LR_Decay: " + lrDecay + ", LR_Step: " + lrStep +
                ", Seed: " + seed);
        System.out.println("[INFO] Dataset split -> Total: " + totalSamples +
                ", Train: " + trainSize + ", Val: " + valSize);

        try {
            Trainer trainer = new Trainer(dataPath, labelPath, batchSize, maxEpochs,
                    patience, initLr, lrDecay, lrStep, seed,
                    totalSamples, trainSize, valSize);
            trainer.prepareData();
            trainer.train(outputPath);
            System.out.println("[INFO] Pipeline execution completed successfully.");
        } catch (Exception e) {
            System.err.println("[ERROR] Pipeline execution failed.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                String key = arg.substring(2);
                if (key.equals("help")) {
                    map.put("help", "true");
                } else if (i + 1 < args.length) {
                    map.put(key, args[++i]);
                }
            } else if (arg.startsWith("-")) {
                String key = arg.substring(1);
                if (key.equals("h")) {
                    map.put("help", "true");
                } else if (i + 1 < args.length) {
                    map.put(key, args[++i]);
                }
            }
        }
        return map;
    }

    private static void printHelp() {
        System.out.println("Usage: java [JVM_ARGS] -cp . com.mc.pbr.training.TrainMain [options]");
        System.out.println("Options:");
        System.out.println("  -d, --data <path>            Path to train_data.bin (default: train_data.bin)");
        System.out.println("  -l, --labels <path>          Path to train_labels.bin (default: train_labels.bin)");
        System.out.println("  -o, --output <path>          Path to save height_model.ser (default: height_model.ser)");
        System.out.println("  -b, --batch-size <int>       Batch size for training (default: 64)");
        System.out.println("  -e, --epochs <int>           Maximum number of epochs (default: 50)");
        System.out.println("  -p, --patience <int>         Early stopping patience (default: 8)");
        System.out.println("  -r, --lr <float>             Initial learning rate (default: 0.01)");
        System.out.println("  -y, --lr-decay <float>       Learning rate decay factor (default: 0.9)");
        System.out.println("  -s, --lr-step <int>          Epochs per learning rate decay (default: 10)");
        System.out.println("  -S, --seed <long>            Random seed (default: 42)");
        System.out.println("  -T, --total-samples <int>    Total number of samples in dataset (default: 2000000)");
        System.out.println("  -t, --train-size <int>       Number of training samples (default: 1000000)");
        System.out.println("  -v, --val-size <int>         Number of validation samples (default: 20000)");
        System.out.println("  -h, --help                   Show this help message");
    }
}