package info.klewitz.kaggle.forest.weka

import weka.classifiers.Evaluation
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import weka.core.SerializationHelper

class WekaBuildRandomForestApp {

  public static void main(String[] args) {

    Instances data = WekaDataUtils.loadData("train.arff")
    data = WekaDataUtils.removeIndices(data, "first")

    String[] options = new String[1];
    options[0] = "-print";            // unpruned tree
    RandomForest tree = new RandomForest();         // new instance of tree

    tree.setOptions(options);     // set the options
    tree.buildClassifier(data);   // build classifier

    println tree.globalInfo()
    println tree
    println tree.printTreesTipText()

    Evaluation eTest = new Evaluation(data);
    eTest.evaluateModel(tree, data);
    String strSummary = eTest.toSummaryString();
    println(strSummary)

    SerializationHelper.write("random_forest_cleaned.model", tree)
  }
}

