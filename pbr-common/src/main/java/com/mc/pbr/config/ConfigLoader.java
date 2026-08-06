package com.mc.pbr.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class ConfigLoader {

    private static final ObjectMapper MAPPER = new ObjectMapper(new YAMLFactory())
            .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    public static PBRConfig load(String[] args) {
        PBRConfig config = new PBRConfig();
        String configPath = null;

        for (int i = 0; i < args.length; i++) {
            if (("--config".equals(args[i]) || "-config".equals(args[i])) && i + 1 < args.length) {
                configPath = args[++i];
                break;
            }
        }

        if (configPath != null) {
            try {
                config = MAPPER.readValue(new File(configPath), PBRConfig.class);
            } catch (Exception e) {
                System.err.println("[ERROR] Failed to load config file: " + configPath);
                e.printStackTrace();
                System.exit(1);
            }
        }

        applyCliOverrides(config, args);
        return config;
    }

    private static void applyCliOverrides(PBRConfig config, String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                String key = arg.substring(2);
                if ("config".equals(key)) { i++; continue; }
                if (i + 1 < args.length) map.put(key, args[++i]);
            } else if (arg.startsWith("-") && arg.length() == 2) {
                String key = arg.substring(1);
                if ("h".equals(key)) continue;
                if (i + 1 < args.length) map.put(key, args[++i]);
            }
        }

        PBRConfig.DatasetBuilderConfig db = config.getDatasetBuilder();
        if (map.containsKey("i")) db.setInputDir(map.get("i"));
        if (map.containsKey("o")) db.setOutputDir(map.get("o"));
        if (map.containsKey("m")) db.setMaxSamples(Integer.parseInt(map.get("m")));

        PBRConfig.TrainingConfig train = config.getTraining();
        if (map.containsKey("d")) train.setDataPath(map.get("d"));
        if (map.containsKey("l")) train.setLabelPath(map.get("l"));
        if (map.containsKey("o")) train.setModelOutput(map.get("o"));
        if (map.containsKey("b")) train.getHyperparams().setBatchSize(Integer.parseInt(map.get("b")));
        if (map.containsKey("e")) train.getHyperparams().setEpochs(Integer.parseInt(map.get("e")));
        if (map.containsKey("p")) train.getHyperparams().setPatience(Integer.parseInt(map.get("p")));
        if (map.containsKey("r")) train.getHyperparams().setLearningRate(Float.parseFloat(map.get("r")));
        if (map.containsKey("y")) train.getHyperparams().setLrDecay(Float.parseFloat(map.get("y")));
        if (map.containsKey("s")) train.getHyperparams().setLrStep(Integer.parseInt(map.get("s")));
        if (map.containsKey("S")) config.getGlobal().setSeed(Long.parseLong(map.get("S")));
        if (map.containsKey("T")) train.setTotalSamples(Integer.parseInt(map.get("T")));
        if (map.containsKey("t")) train.setTrainSplit(Integer.parseInt(map.get("t")));
        if (map.containsKey("v")) train.setValSplit(Integer.parseInt(map.get("v")));
        if (map.containsKey("L")) {
            String[] parts = map.get("L").split(",");
            int[] layers = new int[parts.length];
            for (int i = 0; i < parts.length; i++) layers[i] = Integer.parseInt(parts[i].trim());
            train.getHyperparams().setLayers(layers);
        }

        PBRConfig.InferenceConfig inf = config.getInference();
        if (map.containsKey("i")) inf.setModelPath(map.get("i"));
        if (map.containsKey("o")) inf.setOutputDir(map.get("o"));
        if (map.containsKey("m")) inf.setModelPath(map.get("m"));
        if (map.containsKey("s")) inf.setNormalStrength(Float.parseFloat(map.get("s")));
        if (map.containsKey("p")) inf.setPixelate(Boolean.parseBoolean(map.get("p")));
        if (map.containsKey("sm")) inf.setBaseSmoothness(Float.parseFloat(map.get("sm")));
        if (map.containsKey("me")) inf.setBaseMetallic(Float.parseFloat(map.get("me")));
        if (map.containsKey("invH")) inf.getHeight().setInvert(Boolean.parseBoolean(map.get("invH")));
        if (map.containsKey("invN")) inf.getNormal().setInvertY(Boolean.parseBoolean(map.get("invN")));
        if (map.containsKey("hs")) inf.getHeight().setStrength(Float.parseFloat(map.get("hs")));
        if (map.containsKey("hmin")) inf.getHeight().setMin(Float.parseFloat(map.get("hmin")));
        if (map.containsKey("hmax")) inf.getHeight().setMax(Float.parseFloat(map.get("hmax")));
        if (map.containsKey("hsm")) inf.getHeight().setSmoothRadius(Integer.parseInt(map.get("hsm")));
        if (map.containsKey("np")) inf.getHeight().setNormPercentile(Float.parseFloat(map.get("np")));
    }
}