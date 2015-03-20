package info.klewitz.kaggle.forest.weka

import info.klewitz.kaggle.forest.utils.Utils
import org.springframework.core.io.ClassPathResource
import weka.classifiers.AbstractClassifier
import weka.core.Instance
import weka.core.Instances
import weka.core.SerializationHelper

class WekaLoadClassifierAndPredictApp {

  public static void main(String[] args) {

    def modelName = 'Randomforest_cross.model'

    Object model = SerializationHelper.read(new ClassPathResource(modelName).inputStream)
    AbstractClassifier tree = (AbstractClassifier) model
    println tree

    Instances data = WekaDataUtils.loadData("test.arff")
    data.setClassIndex(data.numAttributes() - 1);
    File predictionFile = new File(modelName + "-predictions.txt")

    def verteilung = [0, 0, 0, 0, 0, 0, 0]

    for (int i = 0; i < data.numInstances(); i++) {
      Instance instance = data.instance(i)
      def predictionVector = tree.distributionForInstance(instance)
      def predictedValue = Utils.getMaxIndex(predictionVector) + 1
      println instance.toString(0) + " : " + predictedValue + " #### " + predictionVector
      predictionFile.append(instance.toString(0) + "," + predictedValue + "\n")
      verteilung[predictedValue - 1]++
    }

    println "Verteilung " + verteilung
  }
}

