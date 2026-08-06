package com.mc.pbr.training;

import com.mc.pbr.config.ConfigLoader;
import com.mc.pbr.config.PBRConfig;

public class TrainMain {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_BOLD = "\u001B[1m";

    public static void main(String[] args) {
        PBRConfig config = ConfigLoader.load(args);

        System.out.println(ANSI_CYAN + ANSI_BOLD);
        System.out.println("███╗   ███╗ ██████╗██████╗  ██████╗  ");
        System.out.println("████╗ ████║██╔════╝██╔══██╗██╔════╝  ");
        System.out.println("██╔████╔██║██║     ██████╔╝██║  ███╗ ");
        System.out.println("██║╚██╔╝██║██║     ██╔═══╝ ██║   ██║ ");
        System.out.println("██║ ╚═╝ ██║╚██████╗██║     ╚██████╔╝ ");
        System.out.println("╚═╝     ╚═╝ ╚═════╝╚═╝      ╚═════╝  ");
        System.out.println(ANSI_RESET);

        PBRConfig.TrainingConfig train = config.getTraining();
        PBRConfig.HyperparamsConfig hp = train.getHyperparams();

        if (train.getTrainSplit() + train.getValSplit() > train.getTotalSamples()) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "train-size + val-size cannot exceed total-samples");
            System.exit(1);
        }

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Pipeline execution started.");
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Data: " + train.getDataPath() + " | " + train.getLabelPath());
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Output: " + train.getModelOutput());

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Hyperparameters:" + ANSI_RESET);
        System.out.printf("  %-12s: %d\n", "BatchSize", hp.getBatchSize());
        System.out.printf("  %-12s: %d\n", "Epochs", hp.getEpochs());
        System.out.printf("  %-12s: %d\n", "Patience", hp.getPatience());
        System.out.printf("  %-12s: %.4f\n", "LR", hp.getLearningRate());
        System.out.printf("  %-12s: %.4f\n", "LR_Decay", hp.getLrDecay());
        System.out.printf("  %-12s: %d\n", "LR_Step", hp.getLrStep());
        System.out.printf("  %-12s: %d\n", "Seed", config.getGlobal().getSeed());
        System.out.printf("  %-12s: ", "Layers");
        for (int i = 0; i < hp.getLayers().length; i++) {
            System.out.print(hp.getLayers()[i] + (i < hp.getLayers().length - 1 ? "," : ""));
        }
        System.out.println();

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Dataset split:" + ANSI_RESET);
        System.out.printf("  %-12s: %d\n", "Total", train.getTotalSamples());
        System.out.printf("  %-12s: %d\n", "Train", train.getTrainSplit());
        System.out.printf("  %-12s: %d\n", "Val", train.getValSplit());

        try {
            Trainer trainer = new Trainer(
                    train.getDataPath(),
                    train.getLabelPath(),
                    hp.getBatchSize(),
                    hp.getEpochs(),
                    hp.getPatience(),
                    hp.getLearningRate(),
                    hp.getLrDecay(),
                    hp.getLrStep(),
                    config.getGlobal().getSeed(),
                    train.getTotalSamples(),
                    train.getTrainSplit(),
                    train.getValSplit(),
                    hp.getLayers()
            );
            trainer.prepareData();
            trainer.train(train.getModelOutput());
            System.out.println(ANSI_GREEN + ANSI_BOLD + "[INFO] " + ANSI_RESET + "Pipeline execution completed successfully.");
        } catch (Exception e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET + "Pipeline execution failed.");
            e.printStackTrace();
            System.exit(1);
        }
    }
}