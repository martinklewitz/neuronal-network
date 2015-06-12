package info.klewitz.kaggle.forest.weka

import weka.classifiers.Evaluation
import weka.classifiers.meta.Bagging
import weka.classifiers.trees.RandomForest
import weka.classifiers.trees.RandomTree
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.Utils

class WekaBuildRandomForestApp {

  public static void main(String[] args) {

    Instances data = WekaDataUtils.loadData("train.arff")
    data = WekaDataUtils.removeIndices(data, "first")

    String[] options = new String[2];
    //options[0] = "-print";            // unpruned tree
    options[0] = "-I";            // unpruned tree
    options[1] = "10";            // unpruned tree
    MyRandomForest tree = new MyRandomForest();         // new instance of tree

    tree.setOptions(options);     // set the options
    tree.buildClassifier(data);   // build classifier

    println tree.globalInfo()
    println tree
    println tree.printTreesTipText()

    Evaluation eTest = new Evaluation(data);
    eTest.evaluateModel(tree, data);
    String strSummary = eTest.toSummaryString();
    println(strSummary)

    SerializationHelper.write("random_forest_cleaned.model", tree)
  }

  private static class MyRandomForest extends  RandomForest{

    public void buildClassifier(Instances data) throws Exception {
      this.getCapabilities().testWithFail(data);
      data = new Instances(data);
      data.deleteWithMissingClass();
      this.m_bagger = new Bagging();
      this.m_bagger.setRepresentCopiesUsingWeights(true);
      RandomTree rTree = new RandomTree();
      this.m_KValue = this.m_numFeatures;
      if(this.m_KValue < 1) {
        this.m_KValue = (int)Utils.log2((double)(data.numAttributes() - 1)) + 1;
      }

      rTree.setKValue(this.m_KValue);
      rTree.setMaxDepth(this.getMaxDepth());
      rTree.setDoNotCheckCapabilities(true);
      rTree.setMinVarianceProp(0.5D)
      rTree.setMinNum(5)
      println rTree.getOptions()
      this.m_bagger.setClassifier(rTree);
      this.m_bagger.setSeed(this.m_randomSeed);
      this.m_bagger.setNumIterations(this.m_numTrees);
      this.m_bagger.setCalcOutOfBag(true);
      this.m_bagger.setNumExecutionSlots(this.m_numExecutionSlots);
      this.m_bagger.buildClassifier(data);
    }

  }
}

