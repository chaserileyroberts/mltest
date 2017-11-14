import tensorflow as tf
import numpy as np

class VariablesChangeException(Exception):
  pass

class RangeException(Exception):
  pass

class DependencyException(Exception):
  pass


def setup(tf_seed=0, np_seed=0, reset_graph=True):
  if reset_graph:
    tf.reset_default_graph()
  if tf_seed is not None:
    tf.set_random_seed(tf_seed)
  if np_seed is not None:
    np.random.seed(np_seed)

def _initalizer_helper(sess_conf, init_op):
  session = tf.Session(config=sess_conf)
  if init_op == None:
    session.run(tf.global_variables_initializer())
  else:
    session.run(init_op)
  return session

def _var_change_helper(
    vars_change, 
    op, 
    sess_conf, 
    scope, 
    var_list, 
    feed_dict,
    init_op):
  session = _initalizer_helper(sess_conf, init_op)
  if scope != "" and var_list is not None:
    raise AssertionError("Error: Can not use scope and var_list as the same time.")
  elif var_list is None:
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  else:
    variables = var_list
  var_names = [v.name for v in variables]
  before_vals = session.run(variables, feed_dict=feed_dict)
  session.run(op, feed_dict=feed_dict)
  after_vals = session.run(variables, feed_dict=feed_dict)
  for a,b,name in zip(after_vals, before_vals, var_names):
    try:
      if vars_change:
        assert (a != b).any()
      else:
        assert (a == b).all()
    except AssertionError:
      raise VariablesChangeException(
          "Error: {} did{} change when it should{} have.".format(
              name, 
              " not" if vars_change else "",
              "" if vars_change else " not"))


def assert_vars_same(
    op, 
    sess_conf=None, 
    scope="", 
    var_list=None, 
    feed_dict=None,
    init_op=None):
  _var_change_helper(False, op, sess_conf, scope, var_list, feed_dict, init_op)

def assert_vars_change(
    op, 
    sess_conf=None, 
    scope="", 
    var_list=None, 
    feed_dict=None,
    init_op=None):
  _var_change_helper(True, op, sess_conf, scope, var_list, feed_dict, init_op)

def assert_any_greater_than(
    tensor, 
    value, 
    sess_conf, 
    feed_dict=None,
    init_op=None):
  session = _initalizer_helper(sess_conf, init_op)
  output = session.run(tensor, feed_dict=feed_dict)
  try:
    assert (output > value).any()
  except:
    raise RangeException(
        "Error: Tensor {} had all values less than {}".format(
            tensor.name, value))

def assert_all_greater_than(
    tensor, 
    value, 
    sess_conf=None, 
    feed_dict=None,
    init_op=None):
  session = _initalizer_helper(sess_conf, init_op)
  output = session.run(tensor, feed_dict=feed_dict)
  try:
    assert (output > value).all()
  except:
    raise RangeException(
        "Error: Tensor {} had some values less than {}".format(
            tensor.name, value))

def assert_any_less_than(
    tensor, 
    value,
    sess_conf,
    feed_dict=None,
    init_op=None):
  session = _initalizer_helper(sess_conf, init_op)
  output = session.run(tensor, feed_dict=feed_dict)
  try:
    assert (output < value).any()
  except:
    raise RangeException(
        "Error: Tensor {} had all values atleast {}".format(
            tensor.name, value))


def assert_all_less_than(
    tensor, 
    value, 
    sess_conf=None, 
    feed_dict=None,
    init_op=None):
  session = _initalizer_helper(sess_conf, init_op)
  output = session.run(tensor, feed_dict=feed_dict)
  try:
    assert (output < value).all()
  except:
    raise RangeException(
        "Error: Tensor {} had some values atleast {}".format(
            tensor.name, value))

def assert_input_dependency(
    train_op, 
    feed_dict, 
    sess_conf=None, 
    init_op=None):
  if feed_dict == None:
    pass # Not sure what to do in this case...
  session = _initalizer_helper(sess_conf, init_op)
  session.run(train_op, feed_dict)
  for i,kv in enumerate(feed_dict.items()):
    new_feed_dict = dict(
        list(feed_dict.items())[:i] 
        + list(feed_dict.items())[i + 1:])
    try:
      session.run(train_op, new_feed_dict)
    except:
      pass
    else:
      raise DependencyException(
          "Input variable {} had no effect during training".format(
              kv[0].name))

def test_suite(
    out_tensor,
    train_op,
    sess_conf=None,
    output_range=None,
    scope="",
    var_list=None,
    feed_dict=None,
    init_op=None,
    test_all_inputs_dependent=True,
    test_other_vars_dont_change=True,
    test_output_range=True):
  """Full set of common tests to run for most ML programs.
  Args:
    out_tensor: Output tensor of your model.
    train_op: Op you call to train the model.
    sess_conf: Session configuration to use.
    output_range: Optional. The range you expect your output to have.
        If None, then we test if the output has both postivie and negative
        values.
    scope: Scope of the variables that are to be trained by the train_op.
        Default is "". Can not be used with var_list.
    var_list: List of variables that will change by train_op. Default is None.
        Can not be used with scope.
    feed_dict: Feed diction to pass whenever out_tensor or train_op is called.
        Default is None.
    init_op: The operation to call to initialize the network. If set to None, 
        we call tf.global_variables_initializer()
    test_all_inputs_dependent: Make sure that train_op depends on all values
        passed in with feed_dict. We require that all inputs to the network
        are with "tf.placeholder". Default to True.
    test_other_vars_dont_change: Whether we check if the other variables in
        the graph don't change when we call train_op. Default to True.
    test_output_range: Whether we do the output range check. Default to True.
  Raises:
    mltest.VariablesChangeException: If a variable does/does not change 
        when it should not/should have.
    mltest.RangeException: If the output range does not conform to what was 
        expected.
    mltest.DependencyException: If the train_op can be called successfully
        without pass in values to the variables in feed_dict.
    tf.errors.InvalidArgumentError: If you are missing a variable that train_op
        depends on in feed_dict.
  """

  # Grab the nessissary variables.
  if var_list is None:
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  else:
    variables = var_list
  other_vars = list(set(tf.trainable_variables()) - set(variables))

  # Run default variable changes test.
  assert_vars_change(train_op, sess_conf=sess_conf, 
      scope=scope, var_list=var_list, feed_dict=feed_dict)
  # Run the 'other variables test'
  if test_other_vars_dont_change:
    assert_vars_same(train_op, sess_conf=sess_conf, 
        var_list=other_vars, feed_dict=feed_dict)
  
  # Run the range tests
  if test_output_range:
    if output_range is None:
      assert_any_greater_than(out_tensor, 0, sess_conf=sess_conf, 
          feed_dict=feed_dict, init_op=init_op)
      assert_any_less_than(out_tensor, 0, sess_conf=sess_conf, 
          feed_dict=feed_dict, init_op=init_op)
    else:
      assert_all_greater_than(out_tensor, output_range[0], 
          sess_conf=sess_conf, feed_dict=feed_dict, init_op=init_op)
      assert_all_less_than(out_tensor, output_range[1],  
          sess_conf=sess_conf, feed_dict=feed_dict, init_op=init_op)

  # Run the dependency tests.
  if test_all_inputs_dependent:
    assert_input_dependency(train_op, feed_dict, sess_conf, init_op)