import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

public class AttributeFilter {
    public static void main(String args[]) throws Exception{
        //load data
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();

        //use a simple fillter to remove a certain attribute
        //set up option to remove 1st attribute;
        String[] option = new String[]{"-R", "1"};
        //create remove object
        Remove remove = new Remove();
        //set the filter options
        remove.setOptions(option);
        //pass the dataset to the filter
        remove.setInputFormat(dataset);
        //apply filter
        Instances newdata = Filter.useFilter(dataset, remove);

        //save dataset to new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newdata);
        saver.setFile(new File("Data/Wind_data_clean_remove.arff"));
        saver.writeBatch();
    }
}
