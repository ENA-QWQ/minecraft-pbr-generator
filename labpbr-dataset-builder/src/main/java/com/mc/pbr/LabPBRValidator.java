package com.mc.pbr;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class LabPBRValidator {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_BOLD = "\u001B[1m";

    private static final double MAX_OVERFLOW_RATIO = 0.05;

    public List<TextureTriple> validateAndCollect(File rootDir) throws IOException {
        List<TextureTriple> validTriples = new ArrayList<>();
        File[] entries = rootDir.listFiles();
        if (entries == null || entries.length == 0) {
            return validTriples;
        }

        Path tempRoot = Files.createTempDirectory("labpbr_");
        tempRoot.toFile().deleteOnExit();

        int barLength = 50;
        int total = entries.length;
        for (int i = 0; i < total; i++) {
            File entry = entries[i];
            int progress = (int) (((double) (i + 1) / total) * 100);
            int filled = (int) ((progress / 100.0) * barLength);
            StringBuilder bar = new StringBuilder();
            for (int k = 0; k < barLength; k++) {
                if (k < filled) bar.append("█");
                else bar.append("_");
            }
            System.out.print("\r" + ANSI_CYAN + "[VALIDATE] " + ANSI_RESET +
                    "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                    progress + "% | Scanning: " + (i + 1) + "/" + total);

            if (entry.isDirectory()) {
                collectFromPackDir(entry, validTriples);
            } else if (entry.getName().toLowerCase().endsWith(".zip")) {
                Path packDir = Files.createDirectory(tempRoot.resolve(entry.getName() + "_dir"));
                try {
                    unzip(entry, packDir.toFile());
                    collectFromPackDir(packDir.toFile(), validTriples);
                } catch (IOException e) {
                    // Silent fail during progress bar update
                }
            }
        }
        System.out.println();
        return validTriples;
    }

    private void collectFromPackDir(File packDir, List<TextureTriple> triples) {
        File blockDir = new File(packDir, "assets/minecraft/textures/block");
        if (!blockDir.isDirectory()) {
            return;
        }

        File[] files = blockDir.listFiles((d, n) -> n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) {
            return;
        }

        Map<String, File> bases = new HashMap<>();
        Map<String, File> normals = new HashMap<>();
        Map<String, File> specs = new HashMap<>();

        for (File f : files) {
            String name = f.getName();
            if (name.endsWith("_n.png")) {
                normals.put(name.substring(0, name.length() - 6), f);
            } else if (name.endsWith("_s.png")) {
                specs.put(name.substring(0, name.length() - 6), f);
            } else if (name.endsWith(".png")) {
                bases.put(name.substring(0, name.length() - 4), f);
            }
        }

        for (String base : bases.keySet()) {
            File baseFile = bases.get(base);
            File normalFile = normals.get(base);
            File specFile = specs.get(base);
            if (normalFile == null || specFile == null) continue;

            if (validateNormalMap(normalFile)) {
                triples.add(new TextureTriple(baseFile, normalFile, specFile));
            }
        }
    }

    private boolean validateNormalMap(File file) {
        try {
            BufferedImage img = ImageIO.read(file);
            if (img == null) return false;
            int w = img.getWidth(), h = img.getHeight();
            int[] pixels = img.getRGB(0, 0, w, h, null, 0, w);
            int overflow = 0;
            for (int p : pixels) {
                double x = (((p >> 16) & 0xFF) / 127.5) - 1.0;
                double y = (((p >> 8) & 0xFF) / 127.5) - 1.0;
                if (x * x + y * y > 1.01) overflow++;
            }
            double ratio = (double) overflow / (w * h);
            return ratio <= MAX_OVERFLOW_RATIO;
        } catch (IOException e) {
            return false;
        }
    }

    private void unzip(File zipFile, File destDir) throws IOException {
        try (ZipFile zip = new ZipFile(zipFile)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();
            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                File outFile = new File(destDir, entry.getName());
                if (entry.isDirectory()) {
                    outFile.mkdirs();
                } else {
                    outFile.getParentFile().mkdirs();
                    try (InputStream in = zip.getInputStream(entry);
                         OutputStream out = new FileOutputStream(outFile)) {
                        in.transferTo(out);
                    }
                }
            }
        }
    }

    public static class TextureTriple {
        public final File base, normal, specular;

        public TextureTriple(File b, File n, File s) {
            this.base = b;
            this.normal = n;
            this.specular = s;
        }
    }
}