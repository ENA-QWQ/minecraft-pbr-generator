package com.mc.pbr.training;

import java.util.HashMap;
import java.util.Map;

public class TrainMain {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_BOLD = "\u001B[1m";

    private static final Map<String, String> ALIAS = new HashMap<>();
    static {
        ALIAS.put("d", "data");
        ALIAS.put("l", "labels");
        ALIAS.put("o", "output");
        ALIAS.put("b", "batch-size");
        ALIAS.put("e", "epochs");
        ALIAS.put("p", "patience");
        ALIAS.put("r", "lr");
        ALIAS.put("y", "lr-decay");
        ALIAS.put("s", "lr-step");
        ALIAS.put("S", "seed");
        ALIAS.put("T", "total-samples");
        ALIAS.put("t", "train-size");
        ALIAS.put("v", "val-size");
        ALIAS.put("L", "layers");
        ALIAS.put("h", "help");
    }

    public static void main(String[] args) {
        System.out.println(ANSI_CYAN + ANSI_BOLD);
        System.out.println("███╗   ███╗ ██████╗██████╗  ██████╗ ");
        System.out.println("████╗ ████║██╔════╝██╔══██╗██╔════╝ ");
        System.out.println("██╔████╔██║██║     ██████╔╝██║  ███╗");
        System.out.println("██║╚██╔╝██║██║     ██╔═══╝ ██║   ██║");
        System.out.println("██║ ╚═╝ ██║╚██████╗██║     ╚██████╔╝");
        System.out.println("╚═╝     ╚═╝ ╚═════╝╚═╝      ╚═════╝ ");
        System.out.println(ANSI_RESET);

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

        String layersStr = params.getOrDefault("layers", "100,32,16,1");
        String[] layerParts = layersStr.split(",");
        int[] layerSizes = new int[layerParts.length];
        for (int i = 0; i < layerParts.length; i++) {
            layerSizes[i] = Integer.parseInt(layerParts[i].trim());
        }

        if (trainSize + valSize > totalSamples) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "train-size + val-size cannot exceed total-samples");
            System.exit(1);
        }

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pipeline execution started.");
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Data: " + dataPath + " | " + labelPath);
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Output: " + outputPath);

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Hyperparameters:" + ANSI_RESET);
        System.out.printf("  %-12s: %d\n", "BatchSize", batchSize);
        System.out.printf("  %-12s: %d\n", "Epochs", maxEpochs);
        System.out.printf("  %-12s: %d\n", "Patience", patience);
        System.out.printf("  %-12s: %.4f\n", "LR", initLr);
        System.out.printf("  %-12s: %.4f\n", "LR_Decay", lrDecay);
        System.out.printf("  %-12s: %d\n", "LR_Step", lrStep);
        System.out.printf("  %-12s: %d\n", "Seed", seed);
        System.out.printf("  %-12s: %s\n", "Layers", layersStr);

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Dataset split:" + ANSI_RESET);
        System.out.printf("  %-12s: %d\n", "Total", totalSamples);
        System.out.printf("  %-12s: %d\n", "Train", trainSize);
        System.out.printf("  %-12s: %d\n", "Val", valSize);

        try {
            Trainer trainer = new Trainer(dataPath, labelPath, batchSize, maxEpochs,
                    patience, initLr, lrDecay, lrStep, seed,
                    totalSamples, trainSize, valSize, layerSizes);
            trainer.prepareData();
            trainer.train(outputPath);
            System.out.println(ANSI_GREEN + ANSI_BOLD + "[INFO] " + ANSI_RESET + "Pipeline execution completed successfully.");
        } catch (Exception e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Pipeline execution failed.");
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
                String shortKey = arg.substring(1);
                String key = ALIAS.getOrDefault(shortKey, shortKey);
                if (key.equals("help")) {
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
        System.out.println("  -L, --layers <string>        Comma-separated layer sizes (default: 100,32,16,1)");
        System.out.println("  -h, --help                   Show this help message");
    }
}