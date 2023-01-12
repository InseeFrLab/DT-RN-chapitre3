from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, Sequential

def define_model_structure(
    input_dim,
    filters,
    kernel_sizes,
    strides_conv,
    padding,
    pool_sizes,
    strides_maxpool,
    dim_dense_layers,
    activations,
):

    nb_conv_layers = len(filters)
    # Initialisation of the model
    model = Sequential()
    # First layer
    model.add(
        Conv2D(
            filters[0],
            kernel_sizes[0],
            strides=strides_conv[0],
            activation=activations[0],
        )
    )
    model.add(MaxPooling2D(pool_size=pool_sizes[0], strides=strides_maxpool[0]))

    # Conv layers
    for conv_lay in range(1, nb_conv_layers):
        model.add(
            Conv2D(
                filters[conv_lay],
                kernel_sizes[conv_lay],
                strides=strides_conv[conv_lay],
                padding=padding,
                activation=activations[conv_lay],
            )
        )
        model.add(
            MaxPooling2D(
                pool_size=pool_sizes[conv_lay], strides=strides_maxpool[conv_lay]
            )
        )

    # Flat layer
    model.add(Flatten())

    # Dense layers
    for dense_lay in range(len(dim_dense_layers)):
        model.add(
            Dense(
                dim_dense_layers[dense_lay],
                activation=activations[nb_conv_layers + dense_lay],
            )
        )
    
    nb_layers = len(filters) * 2 + len(dim_dense_layers) + 1
    input_used = Input(shape=input_dim)
    layers = model.layers[0](input_used)
    for i in range(1, nb_layers):
        layers = model.layers[i](layers)

    return Model(input_used, layers)
