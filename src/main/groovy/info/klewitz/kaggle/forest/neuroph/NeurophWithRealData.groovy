package info.klewitz.kaggle.forest.neuroph

import info.klewitz.kaggle.forest.utils.Normalizer
import org.springframework.core.io.ClassPathResource

class NeurophWithRealData {

  public static void main(String[] args) {
    Date date = new Date()
    def dateName = date.toLocaleString()
    File outFileName = new File(dateName + '-results.txt')
    NeuralNetworkNeurophLearner networkNeurophLearner = new NeuralNetworkNeurophLearner(1000, 0.01d, 0.15d)
    networkNeurophLearner.loadModel(new ClassPathResource('network_300_54_15_7.out').inputStream)
    println 'loaded'
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("test_3.csv").removeRow(0).normalize().getData()
    println 'read data'
    networkNeurophLearner.init(array, 54, 0)
    //def stats = networkNeurophLearner.runCalculation(networkNeurophLearner.dataSet, 400000)
    def stats = networkNeurophLearner.runCalculation(networkNeurophLearner.dataSet, 400000)
    outFileName.append(networkNeurophLearner.getGeneralStats())
    outFileName.append(stats)
    println 'done'
  }
}
