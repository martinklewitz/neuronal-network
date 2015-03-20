package info.klewitz.kaggle.forest.weka

import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.core.Instances
import weka.core.SerializationHelper

class WekaBuildBayesForestApp {

  public static void main(String[] args) {

    Instances data = WekaDataUtils.loadData("train.arff")
    data = WekaDataUtils.removeIndices(data, "first")

    String[] options = new String[1];
    options[0] = "-output-debug-info";            // unpruned tree
    NaiveBayes bayes = new NaiveBayes();         // new instance of tree

    bayes.setOptions(options);     // set the options
    bayes.buildClassifier(data);   // build classifier

    println bayes.globalInfo()
    println bayes

    Evaluation eTest = new Evaluation(data);
    eTest.evaluateModel(bayes, data);
    String strSummary = eTest.toSummaryString();
    println(strSummary)

    SerializationHelper.write("NaiveBayes.model", bayes)
  }
}

