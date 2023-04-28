
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CSV2Arff {
    public static void main(String args[]) throws Exception{
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Data/Wind_data_clean.csv"));
        Instances data = loader.getDataSet();

        //save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        //save as ARFF
        saver.setFile(new File("Data/Wind_data_clean.arff"));
        saver.writeBatch();


    }
}
