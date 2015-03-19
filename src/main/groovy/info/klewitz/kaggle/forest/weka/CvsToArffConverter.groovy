package info.klewitz.kaggle.forest.weka

import org.springframework.core.io.ClassPathResource
import weka.core.Instances
import weka.core.converters.ArffSaver
import weka.core.converters.CSVLoader

class CvsToArffConverter {

  public static void main(String[] args) {
    ClassPathResource inputFile = new ClassPathResource("train.csv")
    CSVLoader csvLoader = new CSVLoader()
    csvLoader.setSource(inputFile.inputStream)
    Instances data = csvLoader.getDataSet()

    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File("train.arff"));
    saver.writeBatch();
  }
}

