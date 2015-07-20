package com.paycasso.document;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Arrays;

public class DBNLWFEval {
    private static ImageLoader loader = new ImageLoader(10, 10);

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
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("model-single.ser"));
        MultiLayerNetwork model = (MultiLayerNetwork) ois.readObject();
        model.init();

        File image = new File(DBNLWFExample.class.getResource("/images/blue.png").toURI());
        INDArray x = loader.asRowVector(image);

        System.out.println(Arrays.toString(model.predict(x)));
        INDArray output = model.output(x);
        System.out.println(output);
    }

}
