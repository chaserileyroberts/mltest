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

See the [medium post](https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d) for an overview of features.

The most basic test is to use all of the default settings of `test_suite()`. This will run all of the tests described in the [medium post.](https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d)
```python
import mltest
import your_model_file
import tensorflow as tf
import numpy as np

def setup():
  mltest.setup()

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
## Setup
The function `mltest.setup()` will automatically reset the default graph and seed all of the random values for tensorflow, numpy, and python's random.

## Variables change/don't change.
You can also specify which variables you expect to train with each training op. A major use case is for GAN training.

```python

def test_descriminator():
  model = Model()
  mltest.test_suite(
    model.descriminator_output,
    model.train_descriminator,
    scope="descriminator")

def test_generator():
  model = Model()
  mltest.test_suite(
    model.generator_output,
    model.train_generator,
    scope="generator")
```

You can also define the variables to test against directly with a list.

```python

def test_list_of_variables():
  model = Model()
  vars_that_should_train = [var1, var2, ...]
  mltest.test_suite(
    model.generator_output,
    model.train_generator,
    var_list=vars_that_should_train)
```

## Logits Range
If your model has a specific output range rather than linear, you can test to make sure that range stays consistent. In this example, we assume our logits has a tanh output, so all of our values should fall between 0 and 1.

```python

def test_range():
  model = Model()
  mltest.test_suite(
    model.logits,
    model.train_op,
    output_range=(0,1))
```

## Input Dependencies
The last test makes sure all of the variables in feed_dict affect the train_op

```python


def test_range():
  model = Model()
  mltest.test_suite(
    model.logits,
    model.train_op,
    feed_dict={
      input_1: [1.0],
      input_2: [2.0]
    })
```

## NaN and Inf Tensors
Badly initalized or normalized layers can cause NaN or Inf ouputs that a developer may not have been prepared to handle. This isn't caught with < or > comparisons, so we must test for each directly. These have been added to the test suite aswell.
```python

def test_nan():
  x = tf.constant(-1.0)
  y = tf.log(x) # This will output a NaN value.
  mltest.assert_never_nan(y)

def test_inf():
  a = tf.constant(1.0)
  b = tf.constant(0.0)
  c = a / b # This will output inf.
  mltest.assert_never_inf(c)
```
There are still some issues related to unfectable tensors, so this will not catch all inf or NaN bugs as of yet.

## Turn off tests
Finally, you can turn off any of these tests if they do not suit your needs.

```python

mltest.test_suite(
    test_all_inputs_dependent=True, # Test that all vars in feed_dict 
                                    # affect train_op.
    test_other_vars_dont_change=True, # Test that all vars outside 
                                      # of scope or var_list do not change.
    test_output_range=True # Test output range of the first argument.
    )
```

## Contributors
Thanks to everyone who has helped!

dekromp 
