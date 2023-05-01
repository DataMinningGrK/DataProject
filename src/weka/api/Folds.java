package weka.api;
import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

public class Folds {
    public static void main(String args[]) throws Exception{
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);

        NaiveBayes nb = new NaiveBayes();

        int seed = 1;
        int folds = 10;

        Random rand = new Random(seed);
        Instances randData = new Instances(dataset);
        randData.randomize(rand);

        if (randData.classAttribute().isNominal())
            randData.stratify(folds);

        for (int n=0; n < folds; n++){
            Evaluation eval = new Evaluation(randData); 

            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            nb.buildClassifier(train);
            eval.evaluateModel(nb, test);

            System.out.println();
            System.out.println(eval.toMatrixString("=== Confusion matrix for fold" + (n+1) + "/" + folds + " ==="));
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



        }
    }
}
