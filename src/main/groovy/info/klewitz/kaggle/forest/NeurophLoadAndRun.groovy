package info.klewitz.kaggle.forest

import org.springframework.core.io.ClassPathResource

class NeurophLoadAndRun {

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File outFileName = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(1000, 0.01d, 0.1d)
    networkNeurophLearner.loadModel(new ClassPathResource('network-15-good_2.out').inputStream)
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()

    Collections.shuffle(array)

    networkNeurophLearner.init(array)
    networkNeurophLearner.learn()

    def stats = networkNeurophLearner.printNetworkStats()
    outFileName.append(networkNeurophLearner.getGeneralStats())
    outFileName.append(stats)
    println networkNeurophLearner.getGeneralStats()
    println stats
    networkNeurophLearner.writeModel(dateName)
  }
}
