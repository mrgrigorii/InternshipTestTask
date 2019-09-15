from tensorflow.python.keras import layers
from tensorflow.python.keras import models


def get_model(img_shape):
    inputs = layers.Input(shape=img_shape)

    encoder0_pool, encoder0 = encoder_block(inputs, 8)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 16)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 32)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 64)

    center = conv_block(encoder3_pool, 128)

    decoder3 = decoder_block(center, encoder3, 64)
    decoder2 = decoder_block(decoder3, encoder2, 32)
    decoder1 = decoder_block(decoder2, encoder1, 16)
    decoder0 = decoder_block(decoder1, encoder0, 8)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model


def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2),
                                     padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


def get_model_test_0(img_shape):
    '''test model without BatchNormalization'''
    inputs = layers.Input(shape=img_shape)

    encoder0_pool, encoder0 = encoder_block_test_0(inputs, 8)
    encoder1_pool, encoder1 = encoder_block_test_0(encoder0_pool, 16)
    encoder2_pool, encoder2 = encoder_block_test_0(encoder1_pool, 32)
    encoder3_pool, encoder3 = encoder_block_test_0(encoder2_pool, 64)

    center = conv_block(encoder3_pool, 128)

    decoder3 = decoder_block_test_0(center, encoder3, 64)
    decoder2 = decoder_block_test_0(decoder3, encoder2, 32)
    decoder1 = decoder_block_test_0(decoder2, encoder1, 16)
    decoder0 = decoder_block_test_0(decoder1, encoder0, 8)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model


def conv_block_test_0(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block_test_0(input_tensor, num_filters):
    encoder = conv_block_test_0(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block_test_0(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2),
                                     padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder
