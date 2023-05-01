/*
 *  How to use WEKA API in Java 
 *  Copyright (C) 2014 
 *  @author Dr Noureddin M. Sadawi (noureddin.sadawi@gmail.com)
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it as you wish ... 
 *  I ask you only, as a professional courtesy, to cite my name, web page 
 *  and my YouTube Channel!
 *  
 */

//import required classes
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Clustering {
	public static void main(String args[]) throws Exception{
		//load dataset
		String dataset = "Data/Wind_data_clean.arff";
		DataSource source = new DataSource(dataset);
		Instances data = source.getDataSet();
		SimpleKMeans model = new SimpleKMeans();
		model.setNumClusters(4);
		model.buildClusterer(data);
		System.out.println(model);

		ClusterEvaluation clsEval = new ClusterEvaluation();
		//load dataset
		String dataset1 = "Data/Wind_data_clean_unknown.arff";
		DataSource source1 = new DataSource(dataset1);
		//get instances object 
		Instances data1 = source1.getDataSet();

		clsEval.setClusterer(model);
		clsEval.evaluateClusterer(data1);
		
		System.out.println("# of clusters: " + clsEval.getNumClusters());

	}
}
