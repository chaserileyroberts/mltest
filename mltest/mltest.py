import tensorflow as tf
import numpy as np
import random
import collections


class VariablesChangeException(Exception):
    pass


class RangeException(Exception):
    pass


class DependencyException(Exception):
    pass


class NaNTensorException(Exception):
    pass


class InfTensorException(Exception):
    pass


def setup(tf_seed=0, np_seed=0, python_seed=0, reset_graph=True):
    """Automatically setup standard testing configuration.
    Resets tensorflow's default graph and sets various seeds.
    Args:
        tf_seed: Seed for tensorflow. If None, no seed is set.
        np_seed: Seed for numpy. If None, no seed is set.
        python_seed: Seed for random module. If None, no seed is set
        reset_graph: Flag to reset the default graph. Default is True.
    """
    if reset_graph:
        tf.reset_default_graph()
    if tf_seed is not None:
        tf.set_random_seed(tf_seed)
    if np_seed is not None:
        np.random.seed(np_seed)
    if python_seed is not None:
        random.seed(python_seed)


def _initalizer_helper(sess_conf, init_op):
    """Initialization helper.
    Args:
        sess_conf: Session configuration.
        init_op: Initialization operation.
            If None, tf.global_variables_initializer() is used.
    """
    session = tf.Session(config=sess_conf)
    if init_op is None:
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
    """Helper function to see if a variable changed.
    Args:
        vargs_change: Boolean. Whether vars should change or not.
        op: Training operation. Used to change the variables.
        sess_conf: Optional session configuration.
        scope: Scope of variables to test. Can not be used with var_list.
        var_list: List of variables to test, can not be used with scope.
        feed_dict: Feed_dict for sess.run()
        init_op: Initialization operation.
    Raises:
        VariablesChangeException: If a variables changes when it
            shouldn't have and visa-versa.
    """
    session = _initalizer_helper(sess_conf, init_op)
    if scope != "" and var_list is not None:
        raise AssertionError(
            "Error: Can not use scope and var_list as the same time.")
    elif var_list is None:
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    else:
        variables = var_list
    var_names = [v.name for v in variables]
    before_vals = session.run(variables, feed_dict=feed_dict)
    session.run(op, feed_dict=feed_dict)
    after_vals = session.run(variables, feed_dict=feed_dict)
    for a, b, name in zip(after_vals, before_vals, var_names):
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
    """Assert variables stay the same.
    Args:
        op: Training operation. Used to change the variables.
        sess_conf: Optional session configuration.
        scope: Scope of variables to test. Can not be used with var_list.
        var_list: List of variables to test, can not be used with scope.
        feed_dict: Feed_dict for sess.run()
        init_op: Initialization operation.
    Raises:
        VariablesChangeException: If a variable changes.
    """
    _var_change_helper(False, op, sess_conf, scope,
                       var_list, feed_dict, init_op)


def assert_vars_change(
    op,
    sess_conf=None,
    scope="",
    var_list=None,
    feed_dict=None,
        init_op=None):
    """Assert variables change.
    Args:
        op: Training operation. Used to change the variables.
        sess_conf: Optional session configuration.
        scope: Scope of variables to test. Can not be used with var_list.
        var_list: List of variables to test, can not be used with scope.
        feed_dict: Feed_dict for sess.run()
        init_op: Initialization operation.
    Raises:
        VariablesChangeException: If a variable doesn't change.
    """
    _var_change_helper(
        True, op, sess_conf, scope, var_list, feed_dict, init_op)


def assert_any_greater_than(
    tensor,
    value,
    sess_conf,
    feed_dict=None,
        init_op=None):
    """Assert any values in the tensor are greater than a given value.
    Args:
        tensor: Tensor to check.
        value: The given value.
        sess_conf: Session configuration.
        feed_dict: Feed_dict to be passed to sess.run().
        init_op: Initialization operation.
    Raises:
        RangeException: If nothing is greater than value in tensor.
    """
    session = _initalizer_helper(sess_conf, init_op)
    output = session.run(tensor, feed_dict=feed_dict)
    try:
        assert (output > value).any()
    except BaseException:
        raise RangeException(
            "Error: Tensor {} had all values less than {}".format(
                tensor.name, value))


def assert_all_greater_than(
    tensor,
    value,
    sess_conf=None,
    feed_dict=None,
        init_op=None):
    """Assert all tensor values are greater than a given value.
    Args:
        tensor: Tensor to check.
        value: The minimum value.
        sess_conf: Session configuration.
        feed_dict: Feed_dict to be passed to sess.run().
        init_op: Initialization operation.
    Raises:
        RangeException: If something in tensor is less than or equal to value.
    """
    session = _initalizer_helper(sess_conf, init_op)
    output = session.run(tensor, feed_dict=feed_dict)
    try:
        assert (output > value).all()
    except BaseException:
        raise RangeException(
            "Error: Tensor {} had some values less than {}".format(
                tensor.name, value))


def assert_any_less_than(
    tensor,
    value,
    sess_conf,
    feed_dict=None,
        init_op=None):
    """Assert any tensor values are less than a given value.
    Args:
        tensor: Tensor to check.
        value: The given value.
        sess_conf: Session configuration.
        feed_dict: Feed_dict to be passed to sess.run().
        init_op: Initialization operation.
    Raises:
        RangeException: If everything in tensor is greater than or
            equal to value.
    """
    session = _initalizer_helper(sess_conf, init_op)
    output = session.run(tensor, feed_dict=feed_dict)
    try:
        assert (output < value).any()
    except BaseException:
        raise RangeException(
            "Error: Tensor {} had all values atleast {}".format(
                tensor.name, value))


def assert_all_less_than(
    tensor,
    value,
    sess_conf=None,
    feed_dict=None,
        init_op=None):
    """Assert all tensor values are less than a given value.
    Args:
        tensor: Tensor to check.
        value: The maximum value.
        sess_conf: Session configuration.
        feed_dict: Feed_dict to be passed to sess.run().
        init_op: Initialization operation.
    Raises:
        RangeException: If anything in tensor is greater than or
            equal to value.
    """
    session = _initalizer_helper(sess_conf, init_op)
    output = session.run(tensor, feed_dict=feed_dict)
    try:
        assert (output < value).all()
    except BaseException:
        raise RangeException(
            "Error: Tensor {} had some values atleast {}".format(
                tensor.name, value))


def assert_input_dependency(
    train_op,
    feed_dict,
    sess_conf=None,
        init_op=None):
    """Assert that the train_op depends on everything in feed_dict.
    This test works by removing one item from the feed_dict at a time
    and asserting that the train_op fails.
    Args:
        train_op: Training operation.
        feed_dict: The feed_dict to be passed to sess.run()
            Make sure to only include tensors that train_op should depend on.
        sess_conf: Session configuration.
        init_op: Initialization operation.
    Raises:
        DependencyException: If train_op doesn't depend on something
            in feed_dict
    """
    if feed_dict is None:
        pass  # Not sure what to do in this case...
    session = _initalizer_helper(sess_conf, init_op)
    session.run(train_op, feed_dict)
    for i, kv in enumerate(feed_dict.items()):
        new_feed_dict = dict(
            list(feed_dict.items())[:i] + list(feed_dict.items())[i + 1:])
        try:
            session.run(train_op, new_feed_dict)
        except BaseException:
            pass
        else:
            raise DependencyException(
                "Input variable {} had no effect during training".format(
                    kv[0].name))


def op_dependencies(target_op):
    """List of tensors that target_op depends on.

    Code modified from example by mrry.
    Args:
        target_op: The operation in question.
    Returns:
        op_list: List of operations that are required to
            run to get the value of target_op.
    """
    queue = collections.deque()
    if isinstance(target_op, tf.Operation):
        queue.append(target_op)
        visited = set([])
    else:
        queue.append(target_op.op)
        visited = set([target_op])
    while queue:
        op = queue.popleft()
        # `op` is not a variable, so search its inputs and obtain
        for op_input in op.inputs:
            if op_input.op not in visited:
                queue.append(op_input.op)
                visited.add(op_input)
    graph = tf.get_default_graph()
    filtered_ops = [op for op in visited if graph.is_fetchable(op)]
    return filtered_ops


def assert_never_nan(logits, feed_dict=None, sess_conf=None, init_op=None):
    """Checks against intermediary nan values.
    Args:
        logits: Output of the model.
        feed_dict: Feed diction required to obtain logits.
        sess_conf: Session configuration.
        init_op: initialization operation.
    Raise:
        NanTesnorException: If any value is ever NaN.
    """
    session = _initalizer_helper(sess_conf, init_op)
    parent_operation = op_dependencies(logits)
    results = session.run(parent_operation, feed_dict=feed_dict)
    # Remove non ops.
    results = [r for r in results if r is not None]
    for i, result in enumerate(results):
        if np.isnan(result).any():
            raise NaNTensorException(
                "Operation {} had a nan value.".format(
                    parent_operation[i].name))


def assert_never_inf(logits, feed_dict=None, sess_conf=None, init_op=None):
    """Checks against intermediary inf values.
    Args:
        logits: Output of the model.
        feed_dict: Feed diction required to obtain logits.
        sess_conf: Session configuration.
        init_op: initialization operation.
    Raise:
        InfTesnorException: If any value is ever inf.
    """
    session = _initalizer_helper(sess_conf, init_op)
    parent_operation = op_dependencies(logits)
    results = session.run(parent_operation, feed_dict=feed_dict)
    for i, result in enumerate(results):
        if np.isinf(result).any():
            raise InfTensorException(
                "Operation {} had an inf value.".format(
                    parent_operation[i].name))


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
        test_output_range=True,
        test_nan_vals=True,
        test_inf_vals=True):
    """Full set of common tests to run for most ML programs.
    Args:
      out_tensor: Output tensor of your model.
      train_op: Op you call to train the model.
      sess_conf: Session configuration to use.
      output_range: Optional. The range you expect your output to have.
          If None, then we test if the output has both positive and negative
          values.
      scope: Scope of the variables that are to be trained by the train_op.
          Default is "". Can not be used with var_list.
      var_list: List of variables that will change by train_op. Default is
          None. Can not be used with scope.
      feed_dict: Feed diction to pass whenever out_tensor or train_op is
          called. Default is None.
      init_op: The operation to call to initialize the network. If set to None,
          we call tf.global_variables_initializer()
      test_all_inputs_dependent: Make sure that train_op depends on all values
          passed in with feed_dict. We require that all inputs to the network
          are with "tf.placeholder". Default to True.
      test_other_vars_dont_change: Whether we check if the other variables in
          the graph don't change when we call train_op. Default to True.
      test_output_range: Whether we do the output range check. Default to True.
    Raises:
      VariablesChangeException: If a variable does/does not change
          when it should not/should have.
      RangeException: If the output range does not conform to what was
          expected.
      DependencyException: If the train_op can be called successfully
          without pass in values to the variables in feed_dict.
      tf.errors.InvalidArgumentError: If you are missing a variable that
          train_op depends on in feed_dict.
    """

    # Grab the nessissary variables.
    if var_list is None:
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
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
                                    sess_conf=sess_conf,
                                    feed_dict=feed_dict, init_op=init_op)
            assert_all_less_than(out_tensor, output_range[1],
                                 sess_conf=sess_conf, feed_dict=feed_dict,
                                 init_op=init_op)

    # Run the dependency tests.
    if test_all_inputs_dependent:
        assert_input_dependency(train_op, feed_dict, sess_conf, init_op)
    if test_nan_vals:
        assert_never_nan(out_tensor, feed_dict, sess_conf, init_op)
        assert_never_nan(train_op, feed_dict, sess_conf, init_op)
    if test_inf_vals:
        assert_never_inf(out_tensor, feed_dict, sess_conf, init_op)
        assert_never_nan(train_op, feed_dict, sess_conf, init_op)
