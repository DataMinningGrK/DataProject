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
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CombineModels {
	public static void main(String[] args) throws Exception {
		//load dataset
		String data = "";
		DataSource source = new DataSource(data);
		//get instances object 
		Instances trainingData = source.getDataSet();
		//set class index .. as the last attribute
		if (trainingData.classIndex() == -1) {
			trainingData.setClassIndex(trainingData.numAttributes() - 1);
		}

		//AdaBoost .. 
		AdaBoostM1 m1 = new AdaBoostM1();
		m1.setClassifier(new DecisionStump());//needs one base-classifier
		m1.setNumIterations(20);
		m1.buildClassifier(trainingData);

		//Bagging .. 
		Bagging bagger = new Bagging();
		bagger.setClassifier(new RandomTree());//needs one base-model
		bagger.setNumIterations(25);
		bagger.buildClassifier(trainingData);		

		//Stacking ..
		Stacking stacker = new Stacking();
		stacker.setMetaClassifier(new J48());//needs one meta-model
		Classifier[] classifiers = {				
				new J48(),
				new NaiveBayes(),
				new RandomForest()
		};
		stacker.setClassifiers(classifiers);//needs one or more models
		stacker.buildClassifier(trainingData);

		//Vote .. 
		Vote voter = new Vote();
		voter.setClassifiers(classifiers);//needs one or more classifiers
		voter.buildClassifier(trainingData);
	}
}
