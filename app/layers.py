"""
Custom L1 Distance layer module, needed for loading the custom-trained Siamese model.

Explanation:
- The L1 distance (or Manhattan distance) measures similarity/difference between two vectors.
- This layer will be called within a Siamese network to compute the absolute difference
  between the input and validation embeddings.
"""

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore

class L1Dist(Layer):
    """
    Custom L1 Distance Layer:
    Calculates the absolute difference between two embeddings.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the L1Dist layer.
        We inherit from tf.keras.layers.Layer with no additional arguments.
        """
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        """
        Perform the L1 distance calculation:
        returns the absolute difference between the input embedding and the validation embedding.
        """
        return tf.math.abs(input_embedding - validation_embedding)