package info.klewitz.kaggle.forest

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

  public void init(List<List<Double>> dataArray) {
    dataSet = new DataSet(54, 7)
    for (int i = 0; i < dataArray.size(); i++) {
      double[] inputs = dataArray[i].subList(0, 54).toArray()
      double[] outputs = dataArray[i].subList(54, 61).toArray()
      //println 'expected ' + outputs
      dataSet.addRow(new DataSetRow(inputs, outputs));
    }
  }

  public String printNetworkStats(DataSet testData) {
    int hits = 0
    def hitStats = [0, 0, 0, 0, 0, 0, 0]
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
    return " hits: " + hits + "/" + testData.rows.size() + " " + " quote: " + hitQuote +
           " Stats: " + Arrays.toString(hitStats) +
           " Error: " + neuralNet.getLearningRule().previousEpochError + '\n'
  }
}
