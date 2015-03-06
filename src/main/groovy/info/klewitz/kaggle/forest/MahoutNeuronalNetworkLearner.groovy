package info.klewitz.kaggle.forest

import org.apache.mahout.classifier.mlp.MultilayerPerceptron
import org.apache.mahout.classifier.mlp.NeuralNetwork

class MahoutNeuronalNetworkLearner {

  public static final int CAT_NUMBER = 54
  public static final int SPLIT_COUNT = 14000
  private List<org.apache.mahout.math.Vector> data
  int numOfClasses = 7

  public static void main(String[] args) {
    MahoutNeuronalNetworkLearner app = new MahoutNeuronalNetworkLearner()
    app.init()
    app.run()
  }

  public void init() {
    DataReader dataReader = new DataReader()
    data = dataReader.readData("train.csv", CAT_NUMBER, numOfClasses)
  }

  public void run() throws IOException {
    int splitPoint = SPLIT_COUNT;
    List<org.apache.mahout.math.Vector> trainingSet = data.subList(0, splitPoint);
    List<org.apache.mahout.math.Vector> testSet = data.subList(splitPoint, data.size());

    NeuralNetwork ann = new MultilayerPerceptron();
    int featureDimension = data.get(0).size() - numOfClasses;
    ann.addLayer(featureDimension, false, "Sigmoid");
    //ann.addLayer(featureDimension * 3 , false, "Sigmoid");
    ann.addLayer(featureDimension * 2, false, "Sigmoid");
    //ann.addLayer(featureDimension , false, "Sigmoid");
    //ann.addLayer(30 , false, "Sigmoid");
    ann.addLayer(numOfClasses, true, "Sigmoid");
    ann.setCostFunction("Minus_Squared")
    ann.setLearningRate(0.4).setMomentumWeight(0.5)//.setRegularizationWeight(0.005);

    int iteration = 20;
    for (int i = 0; i < iteration; ++i) {
      for (org.apache.mahout.math.Vector trainingInstance : trainingSet) {
        ann.trainOnline(trainingInstance);
      }
      org.apache.mahout.math.Vector res = ann.getOutput(trainingSet.get(0).viewPart(0, trainingSet.get(0).size() - numOfClasses));
      println 'iteration ' + i
      println 'vec ' + res.asFormatString()
    }

    int correctInstances = 0;
    for (org.apache.mahout.math.Vector testInstance : testSet) {
      org.apache.mahout.math.Vector res = ann.getOutput(testInstance.viewPart(0, testInstance.size() - numOfClasses));

      println res.asFormatString()

      double[] actualLabels = new double[numOfClasses];
      for (int i = 0; i < numOfClasses; ++i) {
        actualLabels[i] = res.get(i);
      }
      double[] expectedLabels = new double[numOfClasses];
      for (int i = 0; i < numOfClasses; ++i) {
        expectedLabels[i] = testInstance.get(testInstance.size() - numOfClasses + i);
      }

      int predictedIndex = getProdictedLabel(actualLabels)
/*      println "index " + predictedIndex
      println "actualLabels " + actualLabels
      println "expectedLabels "+expectedLabels
      println "testInstance " + testInstance.viewPart(55,6).asFormatString()
      println '-------------------------'
      */
    }

    double accuracy = (double) correctInstances / testSet.size() * 100;

    System.out.printf("Forest DataSet. Classification precision: %d/%d = %f%%\n",
                      correctInstances, testSet.size(), accuracy);
  }

  int getProdictedLabel(double[] labels) {
    double max = 0.0
    int index
    labels.eachWithIndex { entry, int i ->
      if (max < entry) {
        max = entry
        index = i
      }
    }
    return index
  }
}
