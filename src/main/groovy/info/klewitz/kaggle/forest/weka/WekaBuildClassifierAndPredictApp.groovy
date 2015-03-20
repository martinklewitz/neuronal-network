package info.klewitz.kaggle.forest.weka

import info.klewitz.kaggle.forest.utils.Utils
import org.springframework.core.io.ClassPathResource
import weka.classifiers.trees.J48
import weka.core.Instance
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ArffLoader

class WekaBuildClassifierAndPredictApp {

  public static void main(String[] args) {

    ClassPathResource inputFile = new ClassPathResource("train.arff")

    ArffLoader loader = new ArffLoader();
    loader.setSource(inputFile.inputStream);
    Instances testData = loader.getDataSet();
    testData.setClassIndex(testData.numAttributes() - 1);

    String[] options = new String[1];
    options[0] = "-U";            // unpruned tree
    J48 tree = new J48();         // new instance of tree
    //tree.setOptions(options);     // set the options
    tree.buildClassifier(testData);   // build classifier

    //println tree.globalInfo()
    //println tree
    //println tree.binarySplitsTipText()

    ClassPathResource testFile = new ClassPathResource("test.arff")
    loader = new ArffLoader();
    loader.setSource(testFile.inputStream);
    Instances data = loader.getDataSet();
    data.setClassIndex(testData.numAttributes() - 1);
    File predictionFile = new File("predictions2.txt")

    def verteilung = [0, 0, 0, 0, 0, 0, 0]

    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i)
      def predictionVector = tree.distributionForInstance(instance)
      def predictedValue = Utils.getMaxIndex(predictionVector) + 1
      println instance.toString(0) + " : " + predictedValue + " #### " + predictionVector
      predictionFile.append(instance.toString(0) + ", " + predictedValue + "\n")

      verteilung[predictedValue - 1]++
    }

    println "Verteilung " + verteilung
    SerializationHelper.write("weka_model2.out", tree)
  }
}

