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

  public static void main(String[] args) {
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner()
    networkNeurophLearner.init("train.csv")
    networkNeurophLearner.createNetwork()
    networkNeurophLearner.printNetworkStats();
  }

  private void createNetwork() {
    neuralNet = new MultiLayerPerceptron(54, 70, 7);
    neuralNet.getLearningRule().setMaxIterations(20)
    MomentumBackpropagation momentumBackpropagation = (MomentumBackpropagation) neuralNet.getLearningRule()
    momentumBackpropagation.setMomentum(0.3d)
    momentumBackpropagation.setLearningRate(0.15d)
    neuralNet.getLearningRule().addListener(new LearningEventListener() {

      @Override
      void handleLearningEvent(LearningEvent event) {
        if (event.eventType == LearningEventType.EPOCH_ENDED) {
          MomentumBackpropagation backpropagation = (MomentumBackpropagation) event.source
          def iteration = backpropagation.getCurrentIteration()
          println iteration + " total " + backpropagation.totalNetworkError
        }
      }
    })
    neuralNet.learn(dataSet);
  }

  private void init(String fileName) {
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

  public void printNetworkStats() {
    int hits = 0
    def hitStats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
      //println("Output: " + desiredIndex + "-" + outputIndex + " " + networkOutput[outputIndex] );
    }
    println("Hits: " + hits + " " + hits / dataSet.getRows().size())
    println("HitStats: " + Arrays.toString(hitStats))
  }
}
