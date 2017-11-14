# mltest
ML testing framework to simplify writing ML unit tests.

How to use

```python
import mltest
import your_model_file
import numpy as np

def test_mltest_suite():
  x = tf.placeholder(tf.float32, (None, 100))
  model = your_model_file.build_model(x)
  feed_dict = {
      x: np.random.normal(size=(10, 100))
  }
  mltest.test_suite(
      model.prediction,
      model.train_op,
      feed_dict=feed_dict)
```