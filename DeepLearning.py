import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class DeepLearning:
    def __init__(self) -> None:
        pass


    def model(self, training_data: pd.DataFrame, learning_rate: int=0, epochs: int=100, verbose: int=2):
        '''
        Simple model to solve Linear Regression with Gradient Descent optimizer

        training_data: DataFrame to train the model. mMst have it's own inputs and targets columns
        input_size: Equal to the number of variables you have
        output_size: Equal to the number of outputs you've got (for regressions that's usually 1)
        '''

        # Extract inputs/outputs sizes
        input_size = training_data['inputs'].shape[1]
        output_size = training_data['targets'].shape[1]
        
        # Create a stacked layers Model
        model = tf.keras.Sequential([tf.keras.layers.Dense(
                                                            output_size,
                                                            # kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                                            # bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                                                        )
                                    ])

        # Set optimizer as Gradient Descent and learning_rate
        if learning_rate:
            custom_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            custom_optimizer = tf.keras.optimizers.SGD()

        # Set Model optimizer and loss function
        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

        # Fit the model
        model.fit(training_data['inputs'], training_data['targets'], epochs=epochs, verbose=verbose)
        
        return model