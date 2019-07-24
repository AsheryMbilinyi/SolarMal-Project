package com.ashery;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.PrintWriter;

public class variousClassifiers {

    public Instances data,train,test;


    public void initDataSets(String dataset) throws Exception{
        DataSource source=new DataSource(dataset);
        data=source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        int trainSize=(int)Math.round(data.numInstances()*0.8);
        int testSize=data.numInstances()-trainSize;
        train=new Instances(data,0,trainSize);
        test=new Instances(data,trainSize,testSize);
    }


    public Filter applyingReplaceMissingValueFilter() {
        return new ReplaceMissingValues();
    }

    public FilteredClassifier applyingFilterOnClassifier(Filter f,Classifier c){
        FilteredClassifier filteredClassifier=new FilteredClassifier();
        filteredClassifier.setFilter(f);
        filteredClassifier.setClassifier(c);
        return filteredClassifier;
    }

    public static void main(String[] args) throws Exception{
        variousClassifiers variousClassifiers=new variousClassifiers();
        variousClassifiers.initDataSets("Input/malariaDatasetNominal.arff");

        Filter missingValueReplacements=variousClassifiers.applyingReplaceMissingValueFilter();
        Classifier classifier[] = {AdaBoost.getAdaBoost(),BaggingClassifier.getBagging(),
                DecisionTree.getDecisionTree(), NaiveBayesClassifier.getNaiveBayes(),
                RandomForestClassifier.getRandomForest(),SVM.SupportVector()};


        for(int i=0;i<classifier.length;i++){

		Classifier fc=variousClassifiers.applyingFilterOnClassifier(missingValueReplacements,classifier[i]);
		fc.buildClassifier(variousClassifiers.train);
		Evaluation evaluation=new Evaluation(variousClassifiers.train);
		evaluation.evaluateModel(fc,variousClassifiers.test);
		PrintWriter out=new PrintWriter("Output/"+i+".txt");
		out.println(fc.toString());
		out.println(evaluation.toSummaryString("Evaluation Results:",false));
		out.println(evaluation.toMatrixString("===Overall Confusion Matrix\n"));
		out.close();


        }

    }



}
