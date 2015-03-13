package info.klewitz.kaggle.forest.data

import info.klewitz.kaggle.forest.utils.StreamBasedNormalizer

class DataCleaner {

  public static void main(String[] args) {
    StreamBasedNormalizer normalizer = new StreamBasedNormalizer()
    String array = normalizer.read("train.csv", "test_1.csv", "test_2.csv", "test_3.csv")
    //.removeRow(0)
        .spreadIntegerValueAsRows(54, 7)
        .normalize(1, 54)
        .getDataAsCsv()

    File outFile = new File("cleaned_2.csv")
    outFile.append(array)
  }
}
