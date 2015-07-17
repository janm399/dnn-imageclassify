package org.deeplearning4j.deepbelief;

import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class DBNLWFEval {
    private static ImageLoader loader = new ImageLoader(28,28);

    public static void main(String[] args) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("/Users/janmachacek/lfw/model.ser"));
        MultiLayerNetwork model = (MultiLayerNetwork)ois.readObject();

        File image = new File("/Users/janmachacek/lfw/Aretha_Franklin/Aretha_Franklin_0001.jpg");
        INDArray output = model.output(loader.asRowVector(image));
        System.out.println(output);

        for (int i = 0; i < output.columns(); i++) {
            double score = output.getColumn(i).getDouble(0);
            if (score > 2e-4) {
                System.out.println("Input " + i + ", with score " + score);
            }
        }

        DataSetIterator dataIter = new LFWDataSetIterator(1, LFWDataFetcher.NUM_IMAGES);
        DataSet dataSet = dataIter.next();
        System.out.println(dataSet.getLabels());


    }

}
