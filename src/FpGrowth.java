/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


import java.io.IOException;
import java.util.Collection;

public class FpGrowth  {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {

        BufferedReader reader =
                new BufferedReader(new FileReader( "Data/Wind_data_clean.arff"));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        //System.out.println(data.toSummaryString());
        FPGrowth fp = new FPGrowth();

        fp.setMinMetric(0.9);
        fp.setLowerBoundMinSupport(0.01);
        fp.setFindAllRulesForSupportLevel(true);
        System.out.println(fp.getDelta());
        System.out.println(fp.getFindAllRulesForSupportLevel());
        System.out.println(fp.getLowerBoundMinSupport());
        System.out.println(fp.getMaxNumberOfItems());
        System.out.println(fp.getMinMetric());
        System.out.println(fp.getNumRulesToFind());
        System.out.println(fp.getPositiveIndex());
        try{
            fp.buildAssociations(data);

        }catch (Exception e){
            System.out.println("can not build associated rules");
        }
        System.out.println(fp.toString());
    }

}
