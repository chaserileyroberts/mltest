# mltest
ML testing framework to simplify writing ML unit tests.

How to use

```python
import mltest
import your_model_file
import tensorflow as tf
import numpy as np


# Build your test function.
def test_mltest_suite():
  # Make your model input a placeholder.
  x = tf.placeholder(tf.float32, (None, 100))
  # Build your model.
  model = your_model_file.build_model(x)
  # Give it some random input (Be sure to seed it!!).
  feed_dict = {
      x: np.random.normal(size=(10, 100))
  }
  # Run the test suite!
  mltest.test_suite(
      model.prediction,
      model.train_op,
      feed_dict=feed_dict)
```