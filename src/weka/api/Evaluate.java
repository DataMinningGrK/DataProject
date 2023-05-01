package weka.api;
//import classes
import weka.core.Instances; 
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
public class Evaluate{
    public static void main(String args[]) throws Exception{
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        J48 tree = new J48();
        tree.buildClassifier(dataset);
                
        Evaluation eval = new Evaluation(dataset);
        Random rand = new Random(1);
        int folds = 10;

        DataSource source1 = new DataSource("Data/Wind_data_clean.arff");
        Instances testDataset = source1.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes()-1);
        eval.evaluateModel(tree, testDataset);
        System.out.println(eval.toSummaryString("Evaluation result:\n", false));
        
        System.out.println("Correct % = "+eval.pctCorrect());
        System.out.println("Incorrect % = "+eval.pctIncorrect());
        System.out.println("AUC = "+eval.areaUnderROC(1));
        System.out.println("kappa = "+eval.kappa());
        System.out.println("MAE = "+eval.meanAbsoluteError());
        System.out.println("RMSE = "+eval.rootMeanSquaredError());
        System.out.println("RAE = "+eval.relativeAbsoluteError());
        System.out.println("RRSE = "+eval.rootRelativeSquaredError());
        System.out.println("Precision = "+eval.precision(1));
        System.out.println("Recall = "+eval.recall(1));
        System.out.println("fMeasure = "+eval.fMeasure(1));
        System.out.println("Error Rate = "+eval.errorRate());

        System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
    }
}
