import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.File;

public class AttrSelection {
    public static void main(String args[]) throws Exception{
        //load data
        DataSource source = new DataSource("Data/Wind_data_clean.arff");
        Instances dataset = source.getDataSet();

        //Create AtrributeSelection object
        AttributeSelection filter = new AttributeSelection();
        //create evaluator and search algorithm objects
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        //set the algorithm to search backward
        search.setSearchBackwards(true);
        //set the filter to use evaluator and search algorithm
        filter.setEvaluator(eval);
        filter.setSearch(search);
        //specify the dataset
        filter.setInputFormat(dataset);
        //Apply
        Instances newdata = Filter.useFilter(dataset, filter);

        //save dataset to new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newdata);
        saver.setFile(new File("Data/Wind_data_clean_select.arff"));
        saver.writeBatch();
    }
}
