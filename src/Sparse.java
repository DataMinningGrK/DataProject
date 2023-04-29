import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.NonSparseToSparse;

import java.io.File;

public class Sparse {

    public static void main(String args[]) throws Exception{
        //load data
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();

        //Create NonSparesToSparse object to save in sparse ARFF format
        NonSparseToSparse sp = new NonSparseToSparse();

        //specify the dataset
        sp.setInputFormat(dataset);
        //apply
        Instances newdata = Filter.useFilter(dataset, sp);

        //save dataset to new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newdata);
        saver.setFile(new File("Data/Wind_data_clean_sparse.arff"));
        saver.writeBatch();
    }
}
