package com.ashery;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;

public class SVM {
    public static Classifier SupportVector(){
        SMO svm = new SMO();
        return svm;
    }
}
