package info.klewitz.kaggle.forest

import org.neuroph.core.data.DataSet
import org.neuroph.core.data.DataSetRow
import org.neuroph.core.events.LearningEvent
import org.neuroph.core.events.LearningEventListener
import org.neuroph.core.events.LearningEventType
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

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File f = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(10000, 0.02d, 0.12d)
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()
    networkNeurophLearner.init(array)
    [12, 13, 14, 15, 16, 20, 30].forEach {
      networkNeurophLearner.createNetworkAndLearn(54, it, 7)
      def stats = networkNeurophLearner.printNetworkStats()
      f.append(networkNeurophLearner.getGeneralStats())
      f.append(stats)
      println networkNeurophLearner.getGeneralStats()
      println stats
      networkNeurophLearner.writeModel(dateName)
    }
  }

  private String getGeneralStats() {
    def length = neuralNet.layers[1].neurons.length - 1
    return 'hiddenNodes ' + length + ' learning: ' + learningRate + ' momentum ' + momentum + ' iterations ' + iterations + ' \n'
  }

  public writeModel(String filename) {
    def length = neuralNet.layers[1].neurons.length - 1
    neuralNet.save(filename + '-' + length + '.out')
  }

  public void createNetworkAndLearn(int ... networkNodes) {
    neuralNet = new MultiLayerPerceptron(networkNodes);
    MomentumBackpropagation momentumBackpropagation = new MomentumBackpropagation()
    momentumBackpropagation.setMomentum(momentum)
    momentumBackpropagation.setLearningRate(learningRate)
    neuralNet.setLearningRule(momentumBackpropagation)
    momentumBackpropagation.setNeuralNetwork(neuralNet)
    neuralNet.getLearningRule().setMaxIterations(iterations)
    neuralNet.getLearningRule().addListener(new LearningEventListener() {
      @Override
      void handleLearningEvent(LearningEvent event) {
        if (event.eventType == LearningEventType.EPOCH_ENDED) {
          MomentumBackpropagation backpropagation = (MomentumBackpropagation) event.source
          def iteration = backpropagation.getCurrentIteration()
          println iteration + " totalError " + backpropagation.totalNetworkError
        }
      }
    })
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

  public String printNetworkStats() {
    int hits = 0
    def hitStats = [0, 0, 0, 0, 0, 0, 0]
    for (DataSetRow testSetRow : dataSet.getRows()) {
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
    def hitQuote = hits / dataSet.getRows().size()
    return " hits: " + hits + "/" + dataSet.rows.size() + " " + " quote: " + hitQuote +
           " Stats: " + Arrays.toString(hitStats) +
           " Error: " + neuralNet.getLearningRule().previousEpochError + '\n'
  }
}
