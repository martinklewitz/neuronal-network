package info.klewitz.kaggle.forest.utils

import com.google.common.base.Charsets
import com.google.common.collect.Lists
import com.google.common.io.LineProcessor
import com.google.common.io.Resources

class StreamBasedNormalizer {

  public static final String COMMENT = "#"
  public static final String SPLITTER = ","

  private List<List<Double>> rawData = new ArrayList<>()
  private List<String> rawLines
  private Double[] min
  private Double[] max

  public StreamBasedNormalizer read(String... fileNames) {
    fileNames.each {
      rawData.addAll(Resources.readLines(Resources.getResource(it), Charsets.UTF_8, new DoubleLineProcessor()))
    }
    return this
  }

  public StreamBasedNormalizer removeRow(int rowIndex) {
    rawData.each {
      it.remove(rowIndex)
    }
    return this
  }

  public StreamBasedNormalizer normalize(long fromRow, long toRow) {
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

  public StreamBasedNormalizer spreadIntegerValueAsRows(int rowIndex, int spreadOverRows) {
    rawData.each {
      if (it.size() > (rowIndex)) {
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

  public String getDataAsCsv() {
    StringBuffer strungBuffer = new StringBuffer()
    rawData.eachWithIndex { list, int i ->
      strungBuffer.append(list.join(SPLITTER))
      strungBuffer.append('\n')
    }
    return strungBuffer.toString()
  }

  private class DoubleLineProcessor implements LineProcessor<List<List<Double>>> {

    final List<List<Double>> result = Lists.newArrayList();

    public boolean processLine(String line) {
      if (!line.startsWith(COMMENT)) {
        String[] parts = line.split(SPLITTER)
        this.result.add(parts.collect { Double.parseDouble(it) }.toList())
        return true
      }
      return true;
    }

    public List<List<Double>> getResult() {
      return this.result;
    }
  }
}
