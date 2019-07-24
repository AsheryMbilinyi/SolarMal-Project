import weka.classifiers.Classifier;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.meta.Bagging;

public class BaggingClassifier {

    public static Classifier getBagging(){
        Bagging bags = new Bagging();
        bags.setClassifier(new DecisionStump());
        bags.setNumIterations(5);
        return bags;
    }



}
