import pytest
import tensorflow as tf
import numpy as np
from . import mltest

slim = tf.contrib.slim


def _mlp_broken_builder(x):
    return slim.stack(x, slim.fully_connected, [10, 1])


def _mlp_correct_builder(x):
    net = slim.stack(x, slim.fully_connected, [10, 10])
    return slim.fully_connected(net, 1, activation_fn=None)


def setup():
    tf.reset_default_graph()
    tf.set_random_seed(0)
    np.random.seed(0)


def test_mltest_setup_sanity():
    var = tf.Variable(1.0)
    assert len(tf.trainable_variables()) == 1
    mltest.setup()
    assert len(tf.trainable_variables()) == 0


def test_assert_vars_change():
    var = tf.Variable(1.0)
    train_op = tf.train.AdamOptimizer().minimize(var)
    mltest.assert_vars_change(train_op)


def test_assert_vars_change_failure():
    var = tf.Variable(1.0)
    var2 = tf.Variable(1.0, name="error_value")
    train_op = tf.train.AdamOptimizer().minimize(var)
    with pytest.raises(mltest.VariablesChangeException) as excinfo:
        mltest.assert_vars_change(train_op)  # Should raise an error
        assert 'error_value' in str(excinfo.value)


def test_assert_vars_same():
    var = tf.Variable(1.0)
    var2 = tf.Variable(1.0)
    train_op = tf.train.AdamOptimizer().minimize(var)
    mltest.assert_vars_same(train_op, var_list=[var2])  # Should raise an error


def test_assert_vars_same_failure():
    var = tf.Variable(1.0, name="error_value")
    var2 = tf.Variable(1.0)
    train_op = tf.train.AdamOptimizer().minimize(var)
    with pytest.raises(mltest.VariablesChangeException) as excinfo:
        mltest.assert_vars_same(
            train_op,
            var_list=[var])  # Should raise an error
        assert 'error_value' in str(excinfo.value)


def test_suite():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    output = _mlp_correct_builder(x)
    train = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5))
    }
    mltest.test_suite(output, train, feed_dict=feed_dict)


def test_suite_out_range_broken():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    output = _mlp_broken_builder(x)
    train = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5))
    }
    with pytest.raises(mltest.RangeException) as excinfo:
        mltest.test_suite(output, train, feed_dict=feed_dict)
        assert "add all values atleast 0" in str(excinfo.value)


def test_suite_vars_change():
    var = tf.Variable(1.0, name="disconnected_variable")
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    output = _mlp_correct_builder(x)
    train = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5))
    }
    with pytest.raises(mltest.VariablesChangeException) as excinfo:
        mltest.test_suite(output, train, feed_dict=feed_dict)
        assert (
            "disconnected_variable:0 did not change when it should have."
            in str(excinfo.value))


def test_suite_vars_should_change():
    var = tf.Variable(1.0, name="connected_variable")
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    with tf.variable_scope("MLP"):
        output = _mlp_correct_builder(var * x)
        train = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5))
    }
    with pytest.raises(mltest.VariablesChangeException) as excinfo:
        mltest.test_suite(output,
                          train, scope="MLP", feed_dict=feed_dict)
        assert (
            "connected_variable:0 did change when it should not have."
            in str(excinfo.value))


def test_dependency_sanity():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    output = _mlp_correct_builder(x)
    train_op = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5))
    }
    mltest.assert_input_dependency(train_op, feed_dict)


def test_missing_input():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    output = _mlp_correct_builder(x)
    train_op = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {}
    with pytest.raises(tf.errors.InvalidArgumentError):
        mltest.assert_input_dependency(train_op, feed_dict)


def test_extra_input():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    y = tf.placeholder(tf.float32, (None, 5), name="y")
    output = _mlp_correct_builder(x)
    train_op = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5)),
        y: np.random.normal(size=(100, 5))
    }
    with pytest.raises(mltest.DependencyException) as excinfo:
        mltest.assert_input_dependency(train_op, feed_dict)
        assert (
            "Input variable y:0 had no effect during training"
            in excinfo.value)


def test_full_suite_input_dependency():
    x = tf.placeholder(tf.float32, (None, 5), name="x")
    y = tf.placeholder(tf.float32, (None, 5), name="y")
    output = _mlp_correct_builder(x)
    train_op = tf.train.AdamOptimizer().minimize(output)
    feed_dict = {
        x: np.random.normal(size=(100, 5)),
        y: np.random.normal(size=(100, 5))
    }
    with pytest.raises(mltest.DependencyException) as excinfo:
        mltest.test_suite(output, train_op, feed_dict=feed_dict)


def test_dependencies():
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = a * b
    d = c * a
    op_list = mltest.op_dependencies(d)
    assert len(op_list) == 4


def test_nan_bug():
    a = tf.constant(-1.0)
    b = tf.log(a)
    c = b * a
    with pytest.raises(mltest.NaNTensorException) as excinfo:
        mltest.assert_never_nan(c, None, None, None)


def test_inf_bug():
    a = tf.constant(0.0)
    b = tf.constant(1.0)
    c = b / a
    with pytest.raises(mltest.InfTensorException) as excinfo:
        mltest.assert_never_inf(c, None, None, None)
