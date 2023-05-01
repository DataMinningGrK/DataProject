package weka.api;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;

public class ClassifyInstance {
    public static void main(String args[]) throws Exception{
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances trainDataset = source.getDataSet();
        trainDataset.setClassIndex(trainDataset.numAttributes()-1);
        

        int numClasses = trainDataset.numClasses();
        for (int i = 0; i< numClasses; i++){
            String classValue = trainDataset.classAttribute().value(i);
            System.out.println("Class Value " + i + "is" + classValue);
        }
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainDataset);

<<<<<<< HEAD
        DataSource source1 = new DataSource("Data/Wind_data_clean_unknown.arff");
=======
        DataSource source1 = new DataSource("Data/Wind_data_clean.arff");
>>>>>>> origin/main
        Instances testDataset = source1.getDataSet();
        testDataset.setClassIndex(testDataset.numAttributes()-1);

        System.out.println("=================");
        System.out.println("Actual Class, NB Predicted");
        for(int i = 0; i < testDataset.numAttributes(); i++){
            double actualClass = testDataset.instance(i).classValue();
            String actual = testDataset.classAttribute().value((int) actualClass);
            Instance newInst = testDataset.instance(i);
            double predNB = nb.classifyInstance(newInst);
            String predString = testDataset.classAttribute().value((int) predNB);
            System.out.println(actual + ", " + predString);
        }
    }
}