package org.deeplearning4j.deepbelief;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

public class DBNLWFEval {

    public static void main(String[] args) throws Exception {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("/Users/janmachacek/lfw/model.ser"));
        MultiLayerNetwork model = (MultiLayerNetwork)ois.readObject();
    }

}
