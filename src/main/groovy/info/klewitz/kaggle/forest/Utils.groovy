package info.klewitz.kaggle.forest

class Utils {

  static int getMaxIndex(double[] labels) {
    double max = 0.0
    int index
    labels.eachWithIndex { entry, int i ->
      if (max < entry) {
        max = entry
        index = i
      }
    }
    return index
  }
}
