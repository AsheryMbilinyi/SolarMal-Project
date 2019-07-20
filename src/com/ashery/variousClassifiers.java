import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
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


    public Classifier getDecisionTree(){
        J48 DecisionTree=new J48();
        return DecisionTree;
    }

    public Classifier getNaiveBayes() {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.setUseSupervisedDiscretization(true);
        return naiveBayes;
    }
    public Classifier getRandomForest() {
        RandomForest randomForest = new RandomForest();
        return randomForest;
    }

    public Classifier getAdaBoost(){
        AdaBoostM1 m1=new AdaBoostM1();
        m1.setClassifier(new RandomForest());
        m1.setNumIterations(10);
        return m1;
    }


    public Classifier getBagging(){
        Bagging bags=new Bagging();
        bags.setClassifier(new RandomForest());
        bags.setNumIterations(5);
        return bags;
    }



    public static void main(String[] args) throws Exception{
        variousClassifiers variousClassifiers=new variousClassifiers();
        variousClassifiers.initDataSets("Input/v2/dataForPrediction.arff");
        PrintWriter out=new PrintWriter("Output/filteredBagging.txt");
        Filter missingValueReplacements=variousClassifiers.applyingReplaceMissingValueFilter();
        Classifier bagging=variousClassifiers.getBagging();

        Classifier fc=variousClassifiers.applyingFilterOnClassifier(missingValueReplacements,bagging);
        fc.buildClassifier(variousClassifiers.train);
        Evaluation evaluation=new Evaluation(variousClassifiers.train);
        evaluation.evaluateModel(fc,variousClassifiers.test);

        out.println(fc.toString());
        out.println(evaluation.toSummaryString("Evaluation Results:Filtered Bagging",false));
        out.println(evaluation.toMatrixString("===Overall Confusion Matrix\n"));
        out.close();



    }



















}
