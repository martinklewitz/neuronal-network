package info.klewitz.kaggle.forest.neuroph

import info.klewitz.kaggle.forest.utils.Utils
import org.neuroph.core.data.DataSet
import org.neuroph.core.data.DataSetRow
import org.neuroph.nnet.MultiLayerPerceptron
import org.neuroph.nnet.learning.MomentumBackpropagation

class NeuralNetworkNeurophLearner {

  private DataSet dataSet
  private MultiLayerPerceptron neuralNet
  private int iterations
  private double learningRate
  private double momentum

  NeuralNetworkNeurophLearner(int iterations, double learningRate, double momentum) {
    this.iterations = iterations
    this.learningRate = learningRate
    this.momentum = momentum
  }

  public String getGeneralStats() {
    def length = neuralNet.layers[1].neurons.length - 1
    return 'hiddenNodes ' + length + ' learning: ' + learningRate + ' momentum ' + momentum + ' iterations ' + iterations + ' \n'
  }

  public writeModel(String filename) {
    def length = neuralNet.layers[1].neurons.length - 1
    neuralNet.save(filename + '-' + length + '.out')
  }

  public loadModel(InputStream inputStream) {
    neuralNet = (MultiLayerPerceptron) MultiLayerPerceptron.load(inputStream)
    createPropagation()
  }

  public void createNetwork(int ... networkNodes) {
    neuralNet = new MultiLayerPerceptron(networkNodes);
    createPropagation()
  }

  private void createPropagation() {
    MomentumBackpropagation momentumBackpropagation = new MomentumBackpropagation()
    momentumBackpropagation.setMomentum(momentum)
    momentumBackpropagation.setLearningRate(learningRate)
    neuralNet.setLearningRule(momentumBackpropagation)
    momentumBackpropagation.setNeuralNetwork(neuralNet)
    neuralNet.getLearningRule().setMaxIterations(iterations)
    neuralNet.getLearningRule().addListener(new LoggingListener())
  }

  public void learn() {
    neuralNet.learn(dataSet);
  }

  public void init(List<List<Double>> dataArray, Integer dataColumns, Integer outputColumns) {
    dataSet = new DataSet(dataColumns, outputColumns)
    for (int i = 0; i < dataArray.size(); i++) {
      double[] inputs = dataArray[i].subList(0, dataColumns).toArray()
      double[] outputs = dataArray[i].subList(dataColumns, dataColumns + outputColumns).toArray()
      //println 'expected ' + outputs
      dataSet.addRow(new DataSetRow(inputs, outputs));
    }
  }

  public String runTest(DataSet testData) {
    long hits = 0
    def hitStats = [0, 0, 0, 0, 0, 0, 0]
    StringBuilder builder = new StringBuilder()
    for (DataSetRow testSetRow : testData.getRows()) {
      neuralNet.setInput(testSetRow.getInput());
      neuralNet.calculate();
      double[] networkOutput = neuralNet.getOutput();
      def desiredIndex = Utils.getMaxIndex(testSetRow.getDesiredOutput())
      def outputIndex = Utils.getMaxIndex(networkOutput)
      if (desiredIndex == outputIndex) {
        hits++
        hitStats[desiredIndex]++
      }
    }
    def hitQuote = hits / testData.getRows().size()
    builder.append(" hits: " + hits + "/" + testData.rows.size() + " " + " quote: " + hitQuote)
    builder.append(" Stats: " + Arrays.toString(hitStats))
    builder.append(" Error: " + neuralNet.getLearningRule().previousEpochError + '\n')
    return builder.toString()
  }

  public String runCalculation(DataSet testData, long startIndex) {
    long currentIndex = 0
    def hitStats = [0, 0, 0, 0, 0, 0, 0]
    StringBuilder builder = new StringBuilder()
    for (DataSetRow testSetRow : testData.getRows()) {
      neuralNet.setInput(testSetRow.getInput());
      neuralNet.calculate();
      double[] networkOutput = neuralNet.getOutput();
      def outputIndex = Utils.getMaxIndex(networkOutput)
      def dataNumber = startIndex + currentIndex
      def outputKey = outputIndex + 1
      builder.append(dataNumber + ' ,' + outputKey + '\n')
      //builder.append(Arrays.toString(networkOutput) + '\n')
      hitStats[outputIndex]++
      currentIndex++
    }
    println "Stats: " + Arrays.toString(hitStats)
    return builder.toString()
  }
}
