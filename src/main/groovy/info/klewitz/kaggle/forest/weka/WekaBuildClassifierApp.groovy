package info.klewitz.kaggle.forest.weka

import org.springframework.core.io.ClassPathResource
import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ArffLoader

class WekaBuildClassifierApp {

  public static void main(String[] args) {

    ClassPathResource inputFile = new ClassPathResource("train.arff")

    ArffLoader loader = new ArffLoader();
    loader.setSource(inputFile.inputStream);
    Instances data = loader.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);

    String[] options = new String[1];
    options[0] = "-U";            // unpruned tree
    J48 tree = new J48();         // new instance of tree
    tree.setOptions(options);     // set the options
    tree.buildClassifier(data);   // build classifier

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

