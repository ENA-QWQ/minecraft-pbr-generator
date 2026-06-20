package com.mc.pbr.training;

import java.io.*;

public class BinaryChunkReader {

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
            }
        } finally {
            disData.close();
            disLabel.close();
        }
    }
}