from tensorflow.keras import backend as K
import tensorflow as tf

class FBetaMetric():
    """
    Approximates the F-beta AUC with configurable number of approximation steps.

    args: 
        beta:  Desired beta for F-score curve
        steps: Approximation quality, higher number is higher number of bins
    """
    def __init__(self, beta: float = 1, steps: int = 100):
        self.beta       = beta
        self.steps      = steps


    def _recall(self, y_true, y_pred, cutoff):
        """
        Metric that is internally used and only calculates the recall.
        
        args: 
            y_true:     True classes
            y_pred:     Predicted classes
            cutoff:     Minimum activation required for classifying as positive
        """
        
        # Find all the actual positives for which the activation was higher than the specified cutoff
        true_positives = tf.cast(tf.count_nonzero(K.greater_equal(K.clip(y_true * y_pred, 0, 1), cutoff)), tf.float32)

        # Get all actual positives
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        # Calculate the recall
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def _precision(self, y_true, y_pred, cutoff):
        """
        Metric that is internally used and only calculates the precision.
        
        args: 
            y_true:     True classes
            y_pred:     Predicted classes
            cutoff:     Minimum activation required for classifying as positive
        """

        # Find all the actual positives for which the activation was higher than the specified cutoff
        true_positives = tf.cast(tf.count_nonzero(K.greater_equal(K.clip(y_true * y_pred, 0, 1), cutoff)), tf.float32)

        # Get all predicted positives
        predicted_positives = tf.cast(tf.count_nonzero(K.greater_equal(K.clip(y_pred, 0, 1), cutoff)), tf.float32)

        # Calculate the precision
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    

    def f_beta(self, y_true, y_pred):
        """
        Metric that is internally used and only calculates the recall.
        
        args: 
            y_true:     True classes
            y_pred:     Predicted classes
            cutoff:     Minimum activation required for classifying as positive
        """

        # Initialize F-Beta as 0
        f_score = 0

        # Iterate over each sensitivity threshold for the activation
        # (Small number added to prevent calculation errors)
        for cutoff in [(i + 0.0001) / self.steps for i in range(self.steps)]:

            # Calculate the precision and recall for each cutoff
            precision   = self._precision(y_true, y_pred, cutoff)
            recall      = self._recall(y_true, y_pred, cutoff)

            # Calculate the F-score with specified beta for each cutoff and sum
            f_score += (1 + K.square(self.beta))*((precision*recall)/((K.square(self.beta) * precision)+recall+K.epsilon()))

        # Divide by the number of steps to take the mean
        return f_score / self.steps 