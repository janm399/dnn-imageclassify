package org.deeplearning4j.deepbelief;


import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;


/**
 * Created by agibsonccc on 10/2/14.
 **/
public class DBNLWFExample {
    private static Logger log = LoggerFactory.getLogger(DBNLWFExample.class);


    public static void main(String[] args) throws Exception {

        int numSamples = LFWDataFetcher.NUM_IMAGES;
        int batchSize = 1000;
        int iterations = 5;
        int seed = 123;
        int rows = 28;
        int columns = 28;
        int listenerFreq = iterations/5;

        log.info("Load data....");
        DataSetIterator dataIter = new LFWDataSetIterator(batchSize,numSamples,rows,columns);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(dataIter.inputColumns())
                .nOut(dataIter.totalOutcomes())
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-3)
                .list(4)
                .hiddenLayerSizes(600, 250, 200)
                .override(3,new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        while(dataIter.hasNext()) {
            DataSet next = dataIter.next();
            next.normalizeZeroMeanZeroUnitVariance();
            model.fit(next);
        }

        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("/Users/janmachacek/lfw/model.ser"));
        oos.writeObject(model);
        oos.close();

        log.info("Saved model");

//        log.info("Evaluate model....");`
//        dataIter = new LFWDataSetIterator(100,100);
//        DataSet dataSet = dataIter.next();
//        Evaluation eval = new Evaluation();
//        INDArray output = model.output(dataSet.getFeatureMatrix());
//        eval.eval(dataSet.getLabels(), output);
//        log.info(eval.stats());

    }


}
