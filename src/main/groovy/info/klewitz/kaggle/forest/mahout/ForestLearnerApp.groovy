package info.klewitz.kaggle.forest.mahout

import com.google.common.base.Charsets
import com.google.common.collect.Lists
import com.google.common.io.Resources
import org.apache.mahout.classifier.AbstractVectorClassifier
import org.apache.mahout.classifier.sgd.GradientMachine
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression
import org.apache.mahout.classifier.sgd.PassiveAggressive
import org.apache.mahout.classifier.sgd.UniformPrior
import org.apache.mahout.math.DenseVector
import org.apache.mahout.math.Vector

public class ForestLearnerApp {

  public static final int CAT_NUMBER = 54
  public static final int SPLIT_COUNT = 14000
  private HashMap expectedTypes
  private List<String> raw
  private List<Vector> data

  public static void main(String[] args) {
    ForestLearnerApp app = new ForestLearnerApp()
    app.init()
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()));
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()).learningRate(50));
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()).learningRate(2));
    app.run(new PassiveAggressive(8, ForestLearnerApp.CAT_NUMBER));
    app.run(new GradientMachine(ForestLearnerApp.CAT_NUMBER, 10, 54).learningRate(0.1).regularization(0.01));
  }

  public void init() {
    raw = Resources.readLines(Resources.getResource("train.csv"), Charsets.UTF_8)
    data = Lists.newArrayList()
    expectedTypes = new HashMap<>()

    for (String line : this.raw.subList(1, this.raw.size())) {
      String[] values = line.split(",")
      Vector v = new DenseVector(values.size() - 2);

      for (int i = 1; i < values.length - 2; i++) {
        v.set(i, Double.parseDouble(values[i]));
      }
      this.data.add(v);
      v.set(0, Double.parseDouble(values[0]))
      this.expectedTypes.put(values[0], values[CAT_NUMBER + 1])
    }
  }

  public void run(AbstractVectorClassifier learner) {
    train(data, learner)

    int fit = 0
    int nonFit = 0
    for (int i = SPLIT_COUNT; i < data.size(); i++) {
      Vector classificationVector = learner.classify(data.get(i - 1));
      int maxLikelihoodType = classificationVector.maxValueIndex()
      int expectedType = Integer.parseInt(this.expectedTypes.get("" + (i)))
      if (expectedType == maxLikelihoodType) {
        fit++
      }
      else {
        nonFit++
      }
    }
    println 'learner' + learner.class.name + ' fit: ' + fit + ' nonfit: ' + nonFit + ' perc ' + fit / nonFit
  }

  private void train(ArrayList<Vector> data, AbstractVectorClassifier learner) {
    for (int i = 1; i < SPLIT_COUNT; i++) {
      def expectedCategory = Integer.parseInt(expectedTypes.get("" + i))
      def vector = data.get(i)
      learner.train(expectedCategory, vector);
    }
  }
}
