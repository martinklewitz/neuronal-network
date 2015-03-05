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
  public boolean debug = true

  public static void main(String[] args) {
    Date date = new Date()
    File f = new File(date.toGMTString() + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner()
    networkNeurophLearner.init("train.csv")
    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20].forEach {
      networkNeurophLearner.createNetworkAndLearn(54, it, 7)
      def stats = networkNeurophLearner.printNetworkStats()
      f.append(stats)
      println stats
    }
  }

  public void createNetworkAndLearn(int ... networkNodes) {
    neuralNet = new MultiLayerPerceptron(networkNodes);
    neuralNet.getLearningRule().setMaxIterations(300)
    MomentumBackpropagation momentumBackpropagation = (MomentumBackpropagation) neuralNet.getLearningRule()
    momentumBackpropagation.setMomentum(0.25d)
    momentumBackpropagation.setLearningRate(0.06d)
    neuralNet.getLearningRule().addListener(new LearningEventListener() {
      @Override
      void handleLearningEvent(LearningEvent event) {
        if (event.eventType == LearningEventType.EPOCH_ENDED) {
          MomentumBackpropagation backpropagation = (MomentumBackpropagation) event.source
          def iteration = backpropagation.getCurrentIteration()
          if (debug) {
            println iteration + " totalError " + backpropagation.totalNetworkError
          }
        }
      }
    })
    neuralNet.learn(dataSet);
  }

  public void init(String fileName) {
    dataSet = createDataSet(fileName)
  }

  private DataSet createDataSet(String inputFileName) {
    Normalizer normalizer = new Normalizer()
    def array = normalizer.read(inputFileName).removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()
    DataSet dataSet = new DataSet(54, 7);
    for (int i = 0; i < array.size(); i++) {
      double[] inputs = array[i].subList(0, 54).toArray()
      double[] outputs = array[i].subList(54, 61).toArray()
      //println 'expected ' + outputs
      dataSet.addRow(new DataSetRow(inputs, outputs));
    }
    return dataSet;
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
    def length = neuralNet.layers[1].neurons.length - 1
    return "Network: " + length +
           " hits: " + hits + "/" + dataSet.rows.size() + " " + " quote: " + hitQuote +
           " Stats: " + Arrays.toString(hitStats) +
           " Error: " + neuralNet.getLearningRule().previousEpochError + '\n'
  }
}
