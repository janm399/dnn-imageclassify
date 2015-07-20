package com.paycasso.document;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;

public class DBNLWFExample {
    private static Logger log = LoggerFactory.getLogger(DBNLWFExample.class);

    public static void main(String[] args) throws Exception {
        int seed = 123;
        int rows = 28;
        int columns = 28;
        int iterations = 100;

        ImageLoader loader = new ImageLoader(rows, columns);
        log.info("Load data....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())           // a Restricted Boltzmann Machine
                .nIn(rows * columns)        // the number of pixels on the input
                .nOut(10)                   // the total number outcome classes
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-1)
                .iterations(iterations)
                .list(4)
                .hiddenLayerSizes(600, 250, 200)
                .override(3, new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // if DL4J's API were ``setListeners(List<? extends IterationListener> listeners)``,
        // we could have lived without the typecast.
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        // train the model on single input: this overfits the model, but is a starting point...
        File image = new File(DBNLWFExample.class.getResource("/images/blue.png").toURI());
        INDArray x = loader.asRowVector(image);

        model.fit(x, new int[]{0});    // new int[] {0} is our only label with value 0 (the value can be any valid int)

        // save the model
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("model-single.ser"));
        oos.writeObject(model);
        oos.close();

        log.info("Saved model");
    }


}
