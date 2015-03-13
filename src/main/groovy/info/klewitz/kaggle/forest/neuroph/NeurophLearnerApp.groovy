package info.klewitz.kaggle.forest.neuroph

import info.klewitz.kaggle.forest.utils.Normalizer

public class NeurophLearnerApp {

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File f = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(500, 0.02d, 0.10d)
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()
    Collections.shuffle(array)
    networkNeurophLearner.init(array, 54, 7)
    [14, 15, 16].forEach {
      networkNeurophLearner.createNetwork(54, it, 7)
      networkNeurophLearner.learn()

      networkNeurophLearner.init(array, 54, 7)
      def stats = networkNeurophLearner.runTest(networkNeurophLearner.dataSet)
      f.append(networkNeurophLearner.getGeneralStats())
      f.append(stats)
      println networkNeurophLearner.getGeneralStats()
      println stats
      networkNeurophLearner.writeModel(dateName)
    }
  }
}
