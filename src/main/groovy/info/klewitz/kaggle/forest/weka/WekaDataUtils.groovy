package info.klewitz.kaggle.forest.weka

import org.springframework.core.io.ClassPathResource
import weka.core.Instances
import weka.core.converters.ArffLoader
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Remove

class WekaDataUtils {

  public static Instances loadData(String fileName) {
    ClassPathResource inputFile = new ClassPathResource(fileName)
    ArffLoader loader = new ArffLoader();
    loader.setSource(inputFile.inputStream);
    Instances data = loader.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);
    data
  }

  public static Instances removeIndices(Instances data, String indeces) {
    Remove remove = new Remove();
    remove.setAttributeIndices(indeces);
    remove.setInvertSelection(false);
    remove.setInputFormat(data);
    data = Filter.useFilter(data, remove);
    data
  }
}
