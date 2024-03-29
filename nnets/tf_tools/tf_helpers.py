import logging

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import function

from GenericTools.KerasTools.convenience_operations import dynamic_one_hot, dynamic_ones

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)



# this method seems to be quite unstable given the division by probabilities
def pzToSymbol_noArgmax(cumsum, cumsum_exclusive, value_of_interest):
    # determine the token selected (2 steps: xor and token)
    # differentiable xor (instead of tf.logical_xor)
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    abs_sx = tf.abs(signed_xor)
    eps = 1e-5;
    abs_sx = K.clip(abs_sx, 0 + eps, 1e10 - eps)  # hack
    almost_xor = tf.divide(signed_xor, abs_sx)
    almost_xor = tf.add(almost_xor, -1)
    almost_xor = tf.divide(almost_xor, -2)
    oh_symbol = tf.abs(almost_xor)

    # differentiable argmax (instead of tf.argmax)    
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    abs_c_minus_v = tf.abs(c_minus_v)
    eps = 1e-5;
    abs_c_minus_v = K.clip(abs_c_minus_v, 0 + eps, 1e10 - eps)  # hack
    almost_symbol = tf.divide(c_minus_v, abs_c_minus_v)
    almost_symbol = tf.divide(tf.add(almost_symbol, -1), -2)
    almost_symbol = tf.abs(almost_symbol)
    symbol = tf.reduce_sum(almost_symbol, axis=1)
    symbol = tf.expand_dims(symbol, axis=1)

    return symbol, oh_symbol


@function.Defun()
def argmaxPseudoGrad(cumsum, cumsum_exclusive, value_of_interest, grad):
    dE_dz = tf.cast(grad, dtype=tf.float32)
    # dE_dz = tf.expand_dims(dE_dz, axis=1)

    # c_minus_v = tf.subtract(cumsum, value_of_interest)
    # ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    # signed_xor = c_minus_v * ce_minus_c
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    symbol = tf.argmin(signed_xor, axis=1)
    vocab_size = tf.shape(cumsum)[-1]
    oh_symbol = tf.one_hot(symbol, vocab_size)

    # dz_dc_scaled = tf.maximum(1 - signed_xor, 0)   # val_loss: 0.1689
    # dz_dc_scaled = - 10*signed_xor   # worse than when noArgmax
    dz_dc_scaled = oh_symbol

    cumsum_grad = dE_dz * dz_dc_scaled  # tf.zeros_like(cumsum_exclusive) #dE_dz * c_minus_v # * tf.ones_like(cumsum_exclusive)
    cumsum_exclusive_grad = tf.zeros_like(cumsum_exclusive)  # dE_dz * ce_minus_c #tf.zeros_like(cumsum_exclusive)
    value_grad = tf.ones_like(
        value_of_interest)  # dE_dz*tf.ones_like(value_of_interest)   # ones val_loss: 0.1689 | dE_dz*tf.ones_like(value_of_interest) not very good

    return [cumsum_grad,
            cumsum_exclusive_grad,
            value_grad]


# this method seems to be quite unstable given the division by probabilities
@function.Defun(grad_func=argmaxPseudoGrad)
def pzToSymbol_withArgmax(scaled_cumsum, scaled_cumsum_exclusive, value_of_interest):
    c_minus_v = tf.subtract(scaled_cumsum, value_of_interest)
    ce_minus_c = tf.subtract(scaled_cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    symbol = tf.argmin(signed_xor, axis=1)

    # symbol = tf.expand_dims(symbol, axis=1)
    # symbol = tf.cast(symbol, dtype=tf.float32)
    return symbol


@function.Defun()
def onehotPseudoGrad(token, cumsum, grad):
    vocab_size = tf.shape(cumsum)[-1]
    oh_symbol = tf.one_hot(tf.squeeze(tf.cast(token, dtype=tf.int64), axis=1), vocab_size)
    oh_grad = grad * oh_symbol
    return [oh_grad,
            tf.zeros_like(cumsum)]


@function.Defun(grad_func=onehotPseudoGrad)
def onehot_pseudoD(token, cumsum):
    vocab_size = tf.shape(cumsum)[-1]
    oh_symbol = tf.one_hot(tf.squeeze(tf.cast(token, dtype=tf.int64), axis=1), vocab_size)
    return oh_symbol


def pzToSymbol_derivableMock(cumsum, cumsum_exclusive, value_of_interest):
    c_minus_v = tf.subtract(cumsum, value_of_interest)
    ce_minus_c = tf.subtract(cumsum_exclusive, value_of_interest)
    signed_xor = c_minus_v * ce_minus_c
    symbol = tf.reduce_sum(signed_xor, axis=1)

    return [symbol, cumsum]


def pzToSymbolAndZ(inputs):
    one_softmax, unfolding_point, curDim = inputs
    one_softmax = K.expand_dims(one_softmax, axis=1)
    curDim = tf.cast(tf.reduce_mean(curDim), dtype=tf.int64)

    # FIXME: to make sure the layer can work even if passed an input of values 
    # range, probably worth to raise a warning
    eps = .5e-6
    unfolding_point = K.clip(unfolding_point, 0. + eps, 1. - eps)

    expanded_unfolding_point = K.expand_dims(unfolding_point, axis=1)
    vocab_size = tf.shape(one_softmax)[-1]
    lat_dim = tf.shape(unfolding_point)[-1]

    cumsum = K.cumsum(one_softmax, axis=2)
    cumsum = K.squeeze(cumsum, axis=1)
    cumsum_exclusive = tf.cumsum(one_softmax, axis=2, exclusive=True)
    cumsum_exclusive = K.squeeze(cumsum_exclusive, axis=1)

    x = expanded_unfolding_point[:, :, curDim]
    value_of_interest = tf.tile(x, [1, vocab_size])

    # determine the token selected (2 steps: xor and token)
    # differentiable xor (instead of tf.logical_xor)
    token = pzToSymbol_withArgmax(cumsum, cumsum_exclusive, value_of_interest)
    token = tf.expand_dims(token, axis=1)
    token = tf.cast(token, dtype=tf.float32)
    oh_symbol = onehot_pseudoD(token, cumsum)

    # expand dimensions to be able to perform a proper matrix 
    # multiplication after
    oh_symbol = tf.expand_dims(oh_symbol, axis=1)
    cumsum_exclusive = tf.expand_dims(cumsum_exclusive, axis=1)

    # the c_iti value has to be subtracted to the point for the 
    # next round on this dimension                
    c_iti_value = tf.matmul(oh_symbol, cumsum_exclusive, transpose_b=True)
    c_iti_value = tf.squeeze(c_iti_value, axis=1)
    one_hots = dynamic_one_hot(one_softmax, lat_dim, curDim)
    one_hots = tf.squeeze(one_hots, axis=1)

    c_iti = c_iti_value * one_hots
    unfolding_point = tf.subtract(unfolding_point, c_iti)

    # the p_iti value has to be divided to the point for the next
    # round on this dimension                
    one_hots = dynamic_one_hot(one_softmax, lat_dim, curDim)
    one_hots = tf.squeeze(one_hots, axis=1)
    p_iti_value = tf.matmul(oh_symbol, one_softmax, transpose_b=True)
    p_iti_value = K.squeeze(p_iti_value, axis=1)
    p_iti_and_zeros = p_iti_value * one_hots
    ones = dynamic_ones(one_softmax, lat_dim)
    ones = K.squeeze(ones, axis=1)
    p_iti_plus_ones = tf.add(p_iti_and_zeros, ones)
    p_iti = tf.subtract(p_iti_plus_ones, one_hots)

    unfolding_point = tf.divide(unfolding_point, p_iti)

    return [token, unfolding_point]


def showGradientsAndTrainableParams(model):
    logger.info("""
          Test Gradients
          
          """)
    weights = model.trainable_weights  # weight tensors

    grad = tf.gradients(xs=weights, ys=model.output)
    for g, w in zip(grad, weights):
        logger.info(w)
        logger.info('        ', g)

    logger.info("""
          Number of trainable params
          
          """)

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    logger.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logger.info('Trainable params: {:,}'.format(trainable_count))
    logger.info('Non-trainable params: {:,}'.format(non_trainable_count))


def tf_update_bounds_encoder(low_bound, upp_bound, softmax, s_j, tf_curDim):
    vocab_size = tf.shape(softmax)[-1]
    lat_dim = tf.shape(low_bound)[-1]

    s = s_j[:, 0]
    d_oh = tf.one_hot(tf_curDim * tf.ones_like(s), lat_dim)
    _d_oh = tf.subtract(tf.ones(lat_dim), d_oh, name='d_inv_oh')

    c_upp = K.cumsum(softmax, axis=1)
    c_low = tf.cumsum(softmax, axis=1, exclusive=True)
    range_ = upp_bound[:, tf_curDim] - low_bound[:, tf_curDim]

    s_oh = tf.one_hot(s, vocab_size)

    # tf convoluted way to assign a value to a location ,
    # to minimize time, I'll go to the first and fast solution

    # up bound
    upp_update = range_ * tf.reduce_sum(c_upp * s_oh, axis=1)
    updated_upp = tf.add(low_bound[:, tf_curDim], upp_update)[:, tf.newaxis] * d_oh

    upp_bound = tf.add(upp_bound * _d_oh, updated_upp)

    # low bound
    low_update = range_ * tf.reduce_sum(c_low * s_oh, axis=1)
    updated_low = tf.add(low_bound[:, tf_curDim], low_update)[:, tf.newaxis] * d_oh

    low_bound = tf.add(low_bound * _d_oh, updated_low)

    return low_bound, upp_bound
