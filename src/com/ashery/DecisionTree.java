import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class DecisionTree {
    public static Classifier getDecisionTree(){
        J48 decisionTree= new J48();
        return decisionTree;
    }
}
