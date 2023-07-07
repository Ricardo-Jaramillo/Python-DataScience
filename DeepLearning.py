import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import default_rng


# Setting a seed for random numbers
# print(SeedSequence().entropy)
rng = default_rng(122708692400277160069775657973126599887)


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
    

    def split_datasets(self, dataset, split_into={'train': 0.8, 'val': 0.1, 'test': 0.1}, shuffle_buffer_size=0):
        # Make sure the sum fractions gives 100%
        assert sum(split_into.values()) == 1

        dict_datasets = {}
        
        # Shuffle dataset
        if shuffle_buffer_size:
            # Specify seed to always have the same split distribution between runs
            dataset = dataset.shuffle(shuffle_buffer_size, seed=12)
        
        # Split each dataset in a dict
        dataset_size = dataset.cardinality().numpy()

        for type, frac in split_into.items():
            if frac:
                size = int(frac * dataset_size)
                dict_datasets[type] = dataset.take(size)
                dataset = dataset.skip(size)
        
        return dict_datasets

    
    def deep_model(self, datasets, model_structure, batch_size=100, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=5):
        # Prepare batches of datasets
        for key in datasets.keys():
            if key == 'train':
                size = batch_size
            else:
                size = datasets[key].cardinality().numpy()
            
            datasets[key] = datasets[key].batch(size)
        
        # Create the model
        layers = []

        for layer, item in model_structure.items():
            # Check if current layer is the input/output
            if layer in ('Input', 'Output'):
                
                # If the layer is already set
                if isinstance(item, tf.keras.layers.Layer):
                    layers.append(item)
                
                # else create the layer
                elif isinstance(item, tuple):
                    size, activation = item
                    layers.append(tf.keras.layers.Dense(size, activation=activation))
            
            # append hidden layers
            elif layer == 'Hidden':
                # append for each subset of hidden layers
                for depth, width, activation in item:
                    for i in range(depth):
                        layers.append(tf.keras.layers.Dense(width, activation=activation))

        model = tf.keras.Sequential(layers)

        # Set Optimizer and loss
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Fit the model
        model.fit(datasets['train'], epochs=epochs, validation_data=(next(iter(datasets['val']))), verbose=2)

        # Evaluate the model accuracy
        test_loss, test_accuracy = model.evaluate(datasets['test'])
        print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

        return model
    

    def predict(self, model: tf.keras, dataset: pd.DataFrame):
        '''
        Make predictions based on the model created

        model: Tensorflow model
        dataset: npz file with 'inputs' and 'targets', sames structure as the one used for training
        '''
        
        # Make predictions
        predictions = model.predict_on_batch(dataset['inputs']).round(1)
        # print(predictions)

        # plot predictions vs targets
        plt.plot(np.squeeze(predictions), np.squeeze(dataset['targets']))
        plt.xlabel('outputs')
        plt.ylabel('targets')
        plt.show()