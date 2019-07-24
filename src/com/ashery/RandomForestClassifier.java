package com.ashery;


import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

public class RandomForestClassifier {

    public static Classifier getRandomForest() {
        RandomForest randomForest = new RandomForest();
        return randomForest;
    }


}
