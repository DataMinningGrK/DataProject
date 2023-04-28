import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;

public class Arff2Csv {
    public static void main(String args[]) throws Exception{
        // load Arff
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("Data/Wind_data_clean.arff"));
        Instances data = loader.getDataSet();

        //save csv
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        //save as ARFF
        saver.setFile(new File("Data/Wind_data_clean_check.csv"));
        saver.writeBatch();


    }
}
