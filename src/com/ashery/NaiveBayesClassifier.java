package com.ashery;


import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

public class NaiveBayesClassifier {

    public static Classifier getNaiveBayes() {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.setUseSupervisedDiscretization(true);
        return naiveBayes;
    }

}
