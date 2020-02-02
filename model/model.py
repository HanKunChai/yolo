from keras.layers import Conv2D, Add, ZeroPadding2D, Lambda, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce, wraps
from exception import ValueEmptyException
import config as cfg
import keras.backend as K

# load cfg
cell_size = cfg.CELL_SIZE
classes_num = cfg.CLASSES_NUM
b = cfg.B
c = (classes_num + 2 * (4 + 1))
output_size = cell_size * cell_size * c


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    # reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueEmptyException('compose func args empty not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    # LeakyReLU(BatchNormalization(DarknetConv2D))
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def yolo_model(inputs, rate=.5, trainable=True):
    output = darknet_body(inputs)
    output = Flatten()(output)
    output = Dense(512)(output)
    output = LeakyReLU(alpha=.1)(output)
    output = Dense(4096)(output)
    output = LeakyReLU(alpha=.1)(output)
    output = Dropout(rate=rate, trainable=trainable)(output)
    output = Dense(output_size, activation='linear')(output)
    output = Lambda(output_reshape)(output)
    model = Model(inputs, output)
    return model


def output_reshape(x):
    return K.reshape(x, (-1, cell_size, cell_size, c))
