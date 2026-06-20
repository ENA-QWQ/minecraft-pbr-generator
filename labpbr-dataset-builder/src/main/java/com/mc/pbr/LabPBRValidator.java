package com.mc.pbr;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class LabPBRValidator {

    private static final double MAX_OVERFLOW_RATIO = 0.05;

    public List<TextureTriple> validateAndCollect(File rootDir) throws IOException {
        List<TextureTriple> validTriples = new ArrayList<>();
        File[] entries = rootDir.listFiles();
        if (entries == null || entries.length == 0) {
            return validTriples;
        }

        Path tempRoot = Files.createTempDirectory("labpbr_");
        tempRoot.toFile().deleteOnExit();

        for (File entry : entries) {
            if (entry.isDirectory()) {
                System.out.println("[INFO] Found resource pack directory: " + entry.getName());
                collectFromPackDir(entry, validTriples);
            } else if (entry.getName().toLowerCase().endsWith(".zip")) {
                System.out.println("[INFO] Found resource pack zip: " + entry.getName());
                Path packDir = Files.createDirectory(tempRoot.resolve(entry.getName() + "_dir"));
                try {
                    unzip(entry, packDir.toFile());
                    collectFromPackDir(packDir.toFile(), validTriples);
                } catch (IOException e) {
                    System.out.println("[WARNING] Failed to unzip: " + entry.getName());
                }
            }
        }

        return validTriples;
    }

    private void collectFromPackDir(File packDir, List<TextureTriple> triples) {
        File blockDir = new File(packDir, "assets/minecraft/textures/block");
        if (!blockDir.isDirectory()) {
            System.out.println("[WARNING] Standard texture directory not found in pack: " + blockDir.getAbsolutePath());
            return;
        }

        File[] files = blockDir.listFiles((d, n) -> n.toLowerCase().endsWith(".png"));
        if (files == null || files.length == 0) {
            System.out.println("[WARNING] No PNG files in texture directory");
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
            if (ratio > MAX_OVERFLOW_RATIO) {
                return false;
            }
            return true;
        } catch (IOException e) {
            System.out.println("[WARNING] Failed to read normal map: " + file);
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