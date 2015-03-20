package info.klewitz.kaggle.forest.weka

import org.springframework.core.io.ClassPathResource
import weka.core.Instances
import weka.core.converters.ArffLoader

class WekaDataUtils {

  public static Instances loadData(String fileName) {
    ClassPathResource inputFile = new ClassPathResource(fileName)

    ArffLoader loader = new ArffLoader();
    loader.setSource(inputFile.inputStream);
    Instances data = loader.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);
    data
  }
}
