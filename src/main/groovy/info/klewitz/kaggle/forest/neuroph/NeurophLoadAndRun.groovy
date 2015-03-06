package info.klewitz.kaggle.forest.neuroph

import info.klewitz.kaggle.forest.utils.Normalizer
import org.springframework.core.io.ClassPathResource

class NeurophLoadAndRun {

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File outFileName = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(1000, 0.01d, 0.15d)
    networkNeurophLearner.loadModel(new ClassPathResource('network-16-good_6.out').inputStream)
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").removeRow(0).spreadIntegerValueAsRows(54, 7).normalize().getData()

    //Collections.shuffle(array)

    networkNeurophLearner.init(array, 54, 7)
    //networkNeurophLearner.learn()

    def stats = networkNeurophLearner.runTest(networkNeurophLearner.dataSet)
    outFileName.append(networkNeurophLearner.getGeneralStats())
    outFileName.append(stats)
    println networkNeurophLearner.getGeneralStats()
    println stats
  }
}
