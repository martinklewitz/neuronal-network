package info.klewitz.kaggle.forest.weka

import info.klewitz.kaggle.forest.utils.Utils
import org.springframework.core.io.ClassPathResource
import weka.classifiers.trees.J48
import weka.core.Instance
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ArffLoader

class WekaLoadClassifierAndPredictApp {

  public static void main(String[] args) {

    def modelName = 'J48_C0.25_M2.model'

    Object model = SerializationHelper.read(new ClassPathResource(modelName).inputStream)
    J48 tree = (J48) model
    println tree.globalInfo()
    println tree

    ClassPathResource testFile = new ClassPathResource("test.arff")
    ArffLoader loader = new ArffLoader();
    loader.setSource(testFile.inputStream);
    Instances data = loader.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);
    File predictionFile = new File(modelName + "-predictions.txt")

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
  }
}

