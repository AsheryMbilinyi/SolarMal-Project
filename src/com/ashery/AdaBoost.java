import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;

public class AdaBoost {
    public static Classifier getAdaBoost(){
        AdaBoostM1 m1=new AdaBoostM1();
        m1.setClassifier(new DecisionStump());
        m1.setNumIterations(10);
        return m1;
    }


}
