package info.klewitz.kaggle.forest.mahout

import com.google.common.base.Charsets
import com.google.common.collect.Lists
import com.google.common.io.Resources
import org.apache.mahout.math.DenseVector

class DataReader {

  public List<Vector> readData(String fileName, int numCategories, int numOfClasses) {
    List<List<String>> raw = loadRaw(fileName)
    List<Vector> data = Lists.newArrayList()
    HashMap expectedTypes = new HashMap<>()
    for (List<String> values : raw) {
      DenseVector v = new DenseVector(numCategories + numOfClasses);

      for (int i = 1; i < values.size() - 2; i++) {
        v.set(i, Double.parseDouble(values[i]));
      }
      data.add(v);
      def expected = values[numCategories + 1]
      //println values[0] + " " +  expected
      for (int i = 1; i < numOfClasses; i++) {
        v.set(i + numCategories, expected.equals("" + i) ? 1.0 : 0.0)
      }
      //println v.asFormatString()
      expectedTypes.put(values[0], values[numCategories + 1])
    }
    long[] classhist = [0, 0, 0, 0, 0, 0, 0]
    for (String val : expectedTypes.values()) {
      classhist[Integer.parseInt(val) - 1]++
    }
    println 'hist ' + classhist
    return data
  }

  public List<List<String>> loadRaw(String fileName) {
    List<List<String>> raw = new ArrayList<>()
    List<String> rawLines = Resources.readLines(Resources.getResource(fileName), Charsets.UTF_8)
    for (String line : rawLines.subList(1, rawLines.size())) {
      String[] values = line.split(",")
      raw.add(values.toList())
    }
    return raw
  }
}
