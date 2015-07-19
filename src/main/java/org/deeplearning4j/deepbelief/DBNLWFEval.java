package org.deeplearning4j.deepbelief;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class DBNLWFEval {
    private static ImageLoader loader = new ImageLoader(28, 28);

    private static void x(INDArray features) {
        INDArray columnMeans = features.mean(0);
        INDArray columnStds = features.std(0);

        features.subiRowVector(columnMeans);
        features.diviRowVector(columnStds);
    }

    private static int findMax(INDArray array) {
        double max = array.getDouble(0);
        int currMax = 0;
        for (int col = 1; col < array.columns(); col++) {
            if (array.getDouble(col) > max) {
                max = array.getDouble(col);
                currMax = col;
            }
        }
        return currMax;
    }

    public static void main(String[] args) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("/Users/janmachacek/lfw/model2.ser"));
        MultiLayerNetwork model = (MultiLayerNetwork) ois.readObject();

        File image = new File("/Users/janmachacek/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg");
        INDArray x = loader.asRowVector(image);
        x(x);
        System.out.println(x);
        INDArray output = model.output(x);
        System.out.println(output);
        System.out.println(findMax(output));
    }

}
