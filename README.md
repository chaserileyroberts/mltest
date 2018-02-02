# mltest
Machine learning testing framework.

[![Build Status](https://travis-ci.org/Thenerdstation/mltest.svg?branch=master)](https://travis-ci.org/Thenerdstation/mltest)

## How to install

You can either use it directly from pip
```shell
sudo pip install mltest
```

Or you can copy and paste the mltest.py file. Everthing is self contained and opensourced, so feel free to do what you want!


## How to use

See the [medium post](https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d)

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
      input_tensor: np.random.normal(size=(10, 100)),
      label_tensor: np.random.randint((100))
  }
  # Run the test suite!
  mltest.test_suite(
      model.prediction,
      model.train_op,
      feed_dict=feed_dict)
```


### Buy me pizza

- BTC: 1HkrH3PGToX6MiwfULwW5a4Et8ffhp6nY9

- ETH: 0xf3DAd2b40a7621e42FfFDb060d5c07ecd1A148a3

- LTC: Lc7z3mM4HturLoyoFAZHSbKns1YS1j1jaG

- ZEC: t1WRDYKBYF29cad7Ft32SJeyRpKbTRdVT5g

Half of all proceeds will be donated to make a wish foundation.