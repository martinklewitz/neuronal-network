package info.klewitz.kaggle.forest.utils

class Utils {

  static int getMaxIndex(double[] values) {
    double max = 0.0
    int index
    values.eachWithIndex { entry, int i ->
      if (max < entry) {
        max = entry
        index = i
      }
    }
    return index
  }
}
