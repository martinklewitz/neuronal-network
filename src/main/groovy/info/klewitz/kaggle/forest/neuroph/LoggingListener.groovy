package info.klewitz.kaggle.forest.neuroph

import org.neuroph.core.events.LearningEvent
import org.neuroph.core.events.LearningEventListener
import org.neuroph.core.events.LearningEventType
import org.neuroph.nnet.learning.MomentumBackpropagation

class LoggingListener implements LearningEventListener {

  @Override
  void handleLearningEvent(LearningEvent event) {
    if (event.eventType == LearningEventType.EPOCH_ENDED) {
      MomentumBackpropagation backpropagation = (MomentumBackpropagation) event.source
      def iteration = backpropagation.getCurrentIteration()
      println iteration + " totalError " + backpropagation.totalNetworkError
    }
  }
}
