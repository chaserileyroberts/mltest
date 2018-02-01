# mltest
ML testing framework to simplify writing ML unit tests.
How to install

[![Build Status](https://travis-ci.org/Thenerdstation/mltest.svg?branch=master)](https://travis-ci.org/Thenerdstation/mltest)

```shell
sudo pip install mltest
```


How to use

```python
import mltest
import your_model_file
import tensorflow as tf
import numpy as np


# Build your test function.
def test_mltest_suite():
  # Make your model input a placeholder.
  input_tensor = tf.placeholder(tf.float32, (None, 100))
  label_tensor = tf.placeholder(tf.int32, (None))
  # Build your model.
  model = your_model_file.build_model(input_tensor, label_tensor)
  # Give it some random input (Be sure to seed it!!).
  feed_dict = {
      input_tensor: np.random.normal(size=(10, 100))
      labe
  }
  # Run the test suite!
  mltest.test_suite(
      model.prediction,
      model.train_op,
      feed_dict=feed_dict)
```


Buy me pizza
BTC:
1HkrH3PGToX6MiwfULwW5a4Et8ffhp6nY9

ETH:
0xf3DAd2b40a7621e42FfFDb060d5c07ecd1A148a3

LTC:
Lc7z3mM4HturLoyoFAZHSbKns1YS1j1jaG

ZEC:
t1WRDYKBYF29cad7Ft32SJeyRpKbTRdVT5g

Half of all proceeds will be donated to make a wish foundation.