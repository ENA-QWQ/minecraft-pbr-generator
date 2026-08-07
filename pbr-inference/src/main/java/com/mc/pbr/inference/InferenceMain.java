package com.mc.pbr.inference;

import com.mc.pbr.config.ConfigLoader;
import com.mc.pbr.config.PBRConfig;
import java.io.File;

public class InferenceMain {
    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_BOLD = "\u001B[1m";

    public static void main(String[] args) {
        PBRConfig config = ConfigLoader.load(args);
        String backendType = extractBackend(args);

        System.out.println(ANSI_GREEN + ANSI_BOLD);
        System.out.println("██╗   ██╗ ██████╗████╗  ██████╗ ");
        System.out.println("████╗ ████║██╔════╝██╔══██╗██╔════╝ ");
        System.out.println("██╔████╔██║██║     ██████╔╝██║  ███╗");
        System.out.println("██║╚██╔╝██║██║     ██╔═══╝ ██║   ██║");
        System.out.println("██║ ╚═╝ ██║╚██████╗██║     ╚██████╔╝");
        System.out.println("╚═╝     ╚═╝ ╚═════╝╚═╝      ╚═════╝ ");
        System.out.println(ANSI_RESET);

        PBRConfig.InferenceConfig inf = config.getInference();

        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Backend: " + backendType.toUpperCase());
        System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + ANSI_BOLD + "Inference Configuration:" + ANSI_RESET);
        System.out.printf("  %-15s: %s\n", "Output Dir", new File(inf.getOutputDir()).getAbsolutePath());
        System.out.printf("  %-15s: %s\n", "Model", inf.getModelPath());
        System.out.printf("  %-15s: %.2f\n", "Strength", inf.getNormalStrength());
        System.out.printf("  %-15s: [%.2f, %.2f]\n", "Height Range", inf.getHeight().getMin(), inf.getHeight().getMax());

        try {
            long startTime = System.currentTimeMillis();

            System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Loading Height Model...");
            ModelLoader modelLoader = new ModelLoader(inf.getModelPath());

            System.out.println(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Starting inference pipeline...");
            PBRInferenceEngine engine = new PBRInferenceEngine(
                    modelLoader,
                    inf.getNormalStrength(),
                    inf.isPixelate(),
                    inf.getBaseSmoothness(),
                    inf.getBaseMetallic(),
                    inf.getHeight().isInvert(),
                    inf.getNormal().isInvertY(),
                    inf.getHeight().getStrength(),
                    inf.getHeight().getMin(),
                    inf.getHeight().getMax(),
                    inf.getHeight().getSmoothRadius(),
                    inf.getHeight().getNormPercentile(),
                    backendType
            );

            String inputPath = extractInput(args);
            if (inputPath == null) {
                System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET +
                        "Missing required parameter: -input <path>");
                System.exit(1);
                return;
            }

            engine.process(inputPath, inf.getOutputDir());

            long endTime = System.currentTimeMillis();
            System.out.println(ANSI_GREEN + ANSI_BOLD + "[DONE] " + ANSI_RESET +
                    "Pipeline completed in " + (endTime - startTime) + " ms");

        } catch (Exception e) {
            System.err.println(ANSI_RED + ANSI_BOLD + "[ERROR] " + ANSI_RESET +
                    "Inference failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static String extractBackend(String[] args) {
        for (int i = 0; i < args.length; i++) {
            if (("--backend".equals(args[i]) || "-backend".equals(args[i])) && i + 1 < args.length) {
                return args[i + 1];
            }
        }
        return "cpu";
    }

    private static String extractInput(String[] args) {
        for (int i = 0; i < args.length; i++) {
            if (("-i".equals(args[i]) || "--input".equals(args[i])) && i + 1 < args.length) {
                return args[i + 1];
            }
        }
        return null;
    }
}