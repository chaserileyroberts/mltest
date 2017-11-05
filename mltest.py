import tensorflow as tf
import sys

def mltest_setup():
  tf.reset_default_graph()
  tf.set_random_seed(0)
  np.random.seed(0)

def _var_change_helper(vars_change, op, session, scope="", feed_dict=None):
  variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  var_names = [v.name for v in variables]
  before_vals = sess.run(variables, feed_dict=feed_dict)
  sess.run(op, feed_dict=feed_dict)
  after_vals = sess.run(variables, feed_dict=feed_dict)
  for a,b,name in zip(after_encoder, before_encoder, var_names):
    try:
      if vars_change:
        assert (a != b).any()
      else:
        assert (a == b).all()
    except AssertionError:


def variables_dont_change(op, session, scope="", feed_dict=None):
  _var_change_helper(False, op, session, scope, feed_dict)

def variables_change(op, session, scope="", feed_dict=None):
  _var_change_helper(True, op, session, scope, feed_dict)