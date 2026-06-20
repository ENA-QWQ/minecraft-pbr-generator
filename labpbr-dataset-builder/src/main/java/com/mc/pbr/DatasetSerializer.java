package com.mc.pbr;

import java.io.*;

public class DatasetSerializer {

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
        for (float v : label)   labelOut.writeFloat(v);
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

        System.out.printf("[INFO] Streaming write completed. Total samples: %d, feature dim: %d, label dim: %d%n",
                writtenSamples, featureDim, labelDim);
    }
}