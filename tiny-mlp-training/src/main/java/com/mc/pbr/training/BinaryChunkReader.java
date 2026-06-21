package com.mc.pbr.training;

import java.io.*;

public class BinaryChunkReader {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";

    public static void extractSamples(String dataPath, String labelPath,
                                      int[] sortedIndices,
                                      float[] dataOut, float[] labelOut,
                                      int featureDim, int labelDim) throws IOException {
        DataInputStream disData = new DataInputStream(
                new BufferedInputStream(new FileInputStream(dataPath), 65536));
        DataInputStream disLabel = new DataInputStream(
                new BufferedInputStream(new FileInputStream(labelPath), 65536));
        try {
            int sampleIdx = 0;
            int outPos = 0;
            int total = sortedIndices.length;
            int barLength = 50;
            int updateInterval = Math.max(1, total / 100);
            int processed = 0;

            for (int idx : sortedIndices) {
                while (sampleIdx < idx) {
                    disData.skipBytes(featureDim * 4);
                    disLabel.skipBytes(labelDim * 4);
                    sampleIdx++;
                }
                int dataOff = outPos * featureDim;
                for (int i = 0; i < featureDim; i++) {
                    dataOut[dataOff + i] = disData.readFloat();
                }
                int labelOff = outPos * labelDim;
                for (int i = 0; i < labelDim; i++) {
                    labelOut[labelOff + i] = disLabel.readFloat();
                }
                outPos++;
                sampleIdx++;
                processed++;

                if (processed % updateInterval == 0 || processed == total) {
                    int progress = (int) (((double) processed / total) * 100);
                    int filled = (int) ((progress / 100.0) * barLength);
                    StringBuilder bar = new StringBuilder();
                    for (int i = 0; i < barLength; i++) {
                        if (i < filled) bar.append("█");
                        else bar.append("_");
                    }
                    System.out.print("\r" + ANSI_YELLOW + "[EXTRACT] " + ANSI_RESET +
                            "[" + ANSI_GREEN + bar.toString() + ANSI_RESET + "] " +
                            progress + "% | Samples: " + processed + "/" + total);
                    if (processed == total) {
                        System.out.println();
                    }
                }
            }
        } finally {
            disData.close();
            disLabel.close();
        }
    }
}