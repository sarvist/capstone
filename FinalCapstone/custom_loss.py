import tensorflow as tf


# custom loss to bias forward so that it knows forward better
def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    center_penalty = 0.8

    # this mask only applies to the center direction
    # the 2 there is the one hot encoded value i have for center
    mask = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 2), tf.float32)

    # add the center penalty to loss
    loss = loss + center_penalty * mask * loss
    return loss
