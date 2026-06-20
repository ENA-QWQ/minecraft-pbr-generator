package com.mc.pbr.inference;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class InferenceMain {
    public static void main(String[] args) {
        Map<String, String> params = parseArgs(args);
        if (params.containsKey("help")) {
            printHelp();
            return;
        }

        String inputPath = params.get("input");
        if (inputPath == null) {
            System.err.println("[ERROR] Missing required parameter: -input <path>");
            printHelp();
            return;
        }

        String outputDir = params.getOrDefault("outputDir", "./output");
        String modelPath = params.getOrDefault("model", "height_model.ser");
        float strength = Float.parseFloat(params.getOrDefault("strength", "6.0"));
        boolean pixelate = Boolean.parseBoolean(params.getOrDefault("pixelate", "false"));
        float baseSmoothness = Float.parseFloat(params.getOrDefault("smoothness", "0.2"));
        float baseMetallic = Float.parseFloat(params.getOrDefault("metallic", "0.0"));

        boolean invertHeight = Boolean.parseBoolean(params.getOrDefault("invert-height", "true"));
        boolean invertNormalY = Boolean.parseBoolean(params.getOrDefault("invert-normal-y", "false"));
        float heightStrength = Float.parseFloat(params.getOrDefault("height-strength", "1.2"));
        float heightMin = Float.parseFloat(params.getOrDefault("height-min", "0.2"));
        float heightMax = Float.parseFloat(params.getOrDefault("height-max", "1.0"));
        int heightSmoothRadius = Integer.parseInt(params.getOrDefault("height-smooth", "2"));
        float normPercentile = Float.parseFloat(params.getOrDefault("norm-percentile", "2.0"));

        System.out.println("[INFO] LabPBR Inference Pipeline Initialized");
        System.out.println("[INFO] Input: " + inputPath);
        System.out.println("[INFO] Output Dir: " + new File(outputDir).getAbsolutePath());
        System.out.println("[INFO] Params -> Strength: " + strength + ", HeightRange: [" + heightMin + ", " + heightMax + "]");

        try {
            System.out.println("[INFO] Loading Height Model...");
            ModelLoader heightModel = new ModelLoader(modelPath);

            System.out.println("[INFO] Starting inference pipeline...");
            long startTime = System.currentTimeMillis();

            PBRInferenceEngine engine = new PBRInferenceEngine(heightModel, strength,
                    pixelate, baseSmoothness, baseMetallic, invertHeight, invertNormalY,
                    heightStrength, heightMin, heightMax, heightSmoothRadius, normPercentile);
            engine.process(inputPath, outputDir);

            long endTime = System.currentTimeMillis();
            System.out.println("[DONE] Pipeline completed in " + (endTime - startTime) + " ms");

        } catch (Exception e) {
            System.err.println("[ERROR] Inference failed: " + e.getMessage());
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
                if (key.equals("help")) map.put("help", "true");
                else if (i + 1 < args.length) map.put(key, args[++i]);
            } else if (arg.startsWith("-")) {
                String key = arg.substring(1);
                if (key.equals("h")) map.put("help", "true");
                else if (i + 1 < args.length) map.put(key, args[++i]);
            }
        }
        return map;
    }

    private static void printHelp() {
        System.out.println("Usage: java [JVM_ARGS] -jar pbr-inference.jar [options]");
        System.out.println("Options:");
        System.out.println("  -i, --input <path>                (Required) Input PNG texture path");
        System.out.println("  -o, --outputDir <dir>             Output directory (default: ./output)");
        System.out.println("  -m, --model <path>                Height model path (default: height_model.ser)");
        System.out.println("  -s, --strength <float>            Normal map strength (default: 6.0)");
        System.out.println("  -p, --pixelate <bool>             Enable hard-edge pixelation for low-res (default: false)");
        System.out.println("  -sm, --smoothness <float>         Base smoothness value (default: 0.2)");
        System.out.println("  -me, --metallic <float>           Base metallic value (default: 0.0)");
        System.out.println("  -invH, --invert-height <bool>     Invert height map (default: true)");
        System.out.println("  -invN, --invert-normal-y <bool>   Invert normal Y-axis (default: false)");
        System.out.println("  -hs, --height-strength <float>    Height contrast multiplier (default: 1.2)");
        System.out.println("  -hmin, --height-min <float>       Target minimum height to fix dark bias (default: 0.2)");
        System.out.println("  -hmax, --height-max <float>       Target maximum height (default: 1.0)");
        System.out.println("  -hsm, --height-smooth <int>       Height map box blur radius (0=off, 2=5x5 blur) (default: 2)");
        System.out.println("  -np, --norm-percentile <float>    Percentile cutoff for normalization (default: 2.0)");
        System.out.println("  -h, --help                        Show this help message");
    }
}