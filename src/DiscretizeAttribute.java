import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.NonSparseToSparse;

import java.io.File;

public class DiscretizeAttribute {
    public static void main(String args[]) throws Exception{
        //load data
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();

        //set option
        String[] option = new String[4];
        //chosse the number of interval
        option[0] = "-B"; option[1] = "4";
        //choose the range of attribute on which to apply the filter
        option[2] = "-R";
        option[3] = "first-last";

        //apply discretization
        Discretize discretize = new Discretize();
        discretize.setOptions(option);
        discretize.setInputFormat(dataset);
        Instances newdata = Filter.useFilter(dataset, discretize);



        //save dataset to new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newdata);
        saver.setFile(new File("Data/Wind_data_clean_discretize.arff"));
        saver.writeBatch();
    }
}
