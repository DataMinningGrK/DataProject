import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Regression {
    public static void main(String[] args) throws Exception{
        DataSource source = new DataSource("Data/Wind_data_clean_select.arff");
        Instances dataset = source.getDataSet();
        //set class index to the last attribute
        dataset.setClassIndex(dataset.numAttributes()-1);

        //build model
        LinearRegression lr = new LinearRegression();
        lr.buildClassifier(dataset);
        //output model
        System.out.println(lr);

        /*
        //build model
        SMOreg smo = new SMOreg();
        smo.buildClassifier(dataset);
        //output model
        System.out.println(smo);
         */
    }
}
