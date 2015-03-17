package info.klewitz.kaggle.forest.utils

import com.google.common.base.Charsets
import com.google.common.io.Resources

class Normalizer {

  public static final String COMMENT = "#"
  public static final String splitter = ","
  private Double[] min;
  private Double[] max;

  private List<List<Double>> rawData

  public Normalizer read(String fileName) {
    List<String> rawLines = Resources.readLines(Resources.getResource(fileName), Charsets.UTF_8)
    rawData = new ArrayList<>()
    rawLines.each {
      def strings = it.split(splitter)
      if (!strings.first().startsWith(COMMENT)) {
        List<Double> row = new ArrayList<>()
        strings.each { String entry ->
          row.add(Double.valueOf(entry))
        }
        rawData.add(row)
      }
    }
    return this
  }

  public Normalizer removeRow(int rowIndex) {
    rawData.each {
      it.remove(rowIndex)
    }
    return this
  }

  public Normalizer normalize() {
    Double[] min = new Double[rawData[0].size()];
    Double[] max = new Double[rawData[0].size()];
    rawData[0].eachWithIndex { val, int index ->
      min[index] = val
      max[index] = val
    }
    rawData.each {
      it.eachWithIndex { Double entry, int i ->
        if (min[i] > entry) {
          min[i] = entry
        }
        if (max[i] < entry) {
          max[i] = entry
        }
      }
    }
    rawData.each { lists ->
      lists.eachWithIndex { Double entry, int i ->
        double range = Math.abs(min[i] - max[i])
        if (range > 0) {
          if (min[i] < 0) {
            lists[i] = (entry + Math.abs(min[i])) / range
          }
          else {
            lists[i] = (entry - min[i]) / range
          }
        }
        else {
        }
      }
    }
    return this
  }

  public Normalizer normalize(long fromRow, long toRow) {
    def columns = rawData[0].size()
    min = new Double[columns]
    max = new Double[columns]
    rawData[0].eachWithIndex { val, int index ->
      this.min[index] = val
      this.max[index] = val
    }
    rawData.each {
      it.eachWithIndex { Double entry, int i ->
        if (this.min[i] > entry) {
          this.min[i] = entry
        }
        if (this.max[i] < entry) {
          this.max[i] = entry
        }
      }
    }
    rawData.each { lists ->
      lists.eachWithIndex { Double entry, int i ->
        if (i >= fromRow && i <= toRow) {
          double range = Math.abs(this.min[i] - this.max[i])
          if (range > 0) {
            if (this.min[i] < 0) {
              lists[i] = (entry + Math.abs(this.min[i])) / range
            }
            else {
              lists[i] = (entry - this.min[i]) / range
            }
          }
        }
      }
    }
    return this
  }

  public Normalizer spreadIntegerValueAsRows(int rowIndex, int spreadOverRows) {
    rawData.each {
      def value = it.remove(rowIndex)
      int intValue = value.intValue()
      for (int i = 0; i < spreadOverRows; i++) {
        if (intValue == i + 1) {
          it.add(1.0)
        }
        else {
          it.add(0.0)
        }
      }
    }
    return this
  }

  public List<List<Double>> getData() {
    return rawData
  }

  public Double[][] getDataAsArray() {
    Double[][] arrayOfArrays = new Double[rawData.size()][rawData[0].size()];
    rawData.eachWithIndex { list, int i ->
      arrayOfArrays[i] = list.toArray()
    }
    return arrayOfArrays
  }
}
