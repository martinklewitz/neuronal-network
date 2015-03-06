package info.klewitz.kaggle.forest

public class NeurophApp {

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File f = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(10000, 0.02d, 0.12d)
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()
    networkNeurophLearner.init(array)
    [12, 13, 14, 15, 16, 20, 30].forEach {
      networkNeurophLearner.createNetwork(54, it, 7)

      networkNeurophLearner.learn()

      def stats = networkNeurophLearner.printNetworkStats(networkNeurophLearner.dataSet)
      f.append(networkNeurophLearner.getGeneralStats())
      f.append(stats)
      println networkNeurophLearner.getGeneralStats()
      println stats
      networkNeurophLearner.writeModel(dateName)
    }
  }
}
