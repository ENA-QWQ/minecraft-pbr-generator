package com.mc.pbr;

import java.io.*;

public class DatasetSerializer {
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_BOLD = "\u001B[1m";

    private DataOutputStream dataOut;
    private DataOutputStream labelOut;
    private File dataFile, labelFile;
    private int featureDim, labelDim;
    private int writtenSamples = 0;

    public void open(File dataFile, File labelFile, int featureDim, int labelDim) throws IOException {
        this.dataFile = dataFile;
        this.labelFile = labelFile;
        this.featureDim = featureDim;
        this.labelDim = labelDim;
        this.writtenSamples = 0;

        dataOut = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(dataFile)));
        dataOut.writeInt(0);
        dataOut.writeInt(featureDim);

        labelOut = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(labelFile)));
        labelOut.writeInt(0);
        labelOut.writeInt(labelDim);
    }

    public void writeSample(float[] feature, float[] label) throws IOException {
        for (float v : feature) dataOut.writeFloat(v);
        for (float v : label) labelOut.writeFloat(v);
        writtenSamples++;
    }

    public void close() throws IOException {
        dataOut.close();
        labelOut.close();

        try (RandomAccessFile raf = new RandomAccessFile(dataFile, "rw")) {
            raf.writeInt(writtenSamples);
        }
        try (RandomAccessFile raf = new RandomAccessFile(labelFile, "rw")) {
            raf.writeInt(writtenSamples);
        }

        System.out.printf(ANSI_CYAN + "[INFO] " + ANSI_RESET + "Streaming write completed. Total samples: " +
                        ANSI_GREEN + "%d" + ANSI_RESET + ", feature dim: " + ANSI_GREEN + "%d" + ANSI_RESET +
                        ", label dim: " + ANSI_GREEN + "%d%n" + ANSI_RESET,
                writtenSamples, featureDim, labelDim);
    }
}