package info.klewitz.kaggle.forest.mahout

import com.google.common.collect.Lists
import info.klewitz.kaggle.forest.utils.Normalizer
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
  private HashMap<String, Double> expectedTypes
  private List<Vector> data

  public static void main(String[] args) {
    ForestLearnerApp app = new ForestLearnerApp()
    app.init()
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()));
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()).learningRate(0.05));
    app.run(new OnlineLogisticRegression(8, ForestLearnerApp.CAT_NUMBER, new UniformPrior()).learningRate(2));
    app.run(new PassiveAggressive(8, ForestLearnerApp.CAT_NUMBER));
    app.run(new GradientMachine(ForestLearnerApp.CAT_NUMBER, 10, 54).learningRate(0.1).regularization(0.01));
  }

  public void init() {
    Normalizer normalizer = new Normalizer()
    List<List<Double>> array = normalizer.read("train.csv").normalize(1, 53).getData()
    data = Lists.newArrayList()
    expectedTypes = new HashMap<>()

    for (List<Double> line : array) {
      Vector v = new DenseVector(line.size() - 2);
      for (int i = 0; i < line.size() - 2; i++) {
        v.set(i, line.get(i + 1))
      }
      data.add(v);
      expectedTypes.put(line.get(0).intValue().toString(), line.get(55))
    }
  }

  public void run(AbstractVectorClassifier learner) {
    train(data, learner)

    int fit = 0
    int nonFit = 0
    for (int i = SPLIT_COUNT; i < data.size(); i++) {
      Vector classificationVector = learner.classify(data.get(i - 1));
      int maxLikelihoodType = classificationVector.maxValueIndex()
      int expectedType = expectedTypes.get("" + i).intValue()
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
      Double get = expectedTypes.get("" + i)
      def expectedCategory = get.intValue()
      def vector = data.get(i)
      learner.train(expectedCategory, vector);
    }
  }
}
