package info.klewitz.kaggle.forest.weka

import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.core.Instances
import weka.core.SerializationHelper

class WekaBuildClassifierApp {

  public static void main(String[] args) {

    Instances data = WekaDataUtils.loadData("train.arff")
    data = WekaDataUtils.removeIndices(data, "first")

    String[] options = new String[1];
    options[0] = "-U";
    J48 tree = new J48();
    tree.setOptions(options);
    tree.buildClassifier(data);

    println tree.globalInfo()
    println tree
    println tree.binarySplitsTipText()

    Evaluation eTest = new Evaluation(data);
    eTest.evaluateModel(tree, data);
    String strSummary = eTest.toSummaryString();
    println(strSummary)

    SerializationHelper.write("weka_model.out", tree)
  }
}

