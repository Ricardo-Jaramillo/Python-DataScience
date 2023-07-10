import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.random import default_rng
from DeepLearning import DeepLearning
import matplotlib.pyplot as plt
from SQLServer import SQLServer
import pandas as pd
import numpy as np
import os


# Init SQLServer connection and get data
Therabody = SQLServer('DbTherabody')
query = '''
    select
        Emp,
        agent_name,
        
        Date_Created,
        Date_Closed,
        Date_LastModified,
        Date_FirstResponseToCustomer,
        
        Case_Number,
        Case_RecordType,
        Case_Status,
        Case_Origin,
        Case_OriginAbs,
        Case_CSAT,
        Case_Disposition,
        Case_DispositionReason,
        Case_Disposition_Detailed,
        Case_Product,
        
        Case_FirstResponseToCustomerSeconds / 3600 Case_FRHours,
        Case_HandleTimeHours,
        Case_FRBusinessHours,
        1 as freq

    from V_Case
    where Date_Created >= '2023-06-01'
'''

# Save DataFrames as npz file
# np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
path = os.path.join(os.path.abspath(os.getcwd()), 'datasets')

# Reach data
training_data = np.load(f'{path}/TF_intro.npz')
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# Set Datasets
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)


# Init DeepLearning class
deep_learning = DeepLearning()

# Create a simple linear model
# model = deep_learning.model(training_data=training_data, learning_rate=0.01, epochs=100, verbose=0)

# Make predictions
# deep_learning.predict(model=model, dataset=training_data)

# Create a deep Neural Network
# Split dataset
# dict_datasets_frac = {'train': 0.9, 'val': 0.1}
# BUFFER_SIZE = 10000

# datasets = deep_learning.split_datasets(dataset=scaled_train_and_validation_data, split_into=dict_datasets_frac, shuffle_buffer_size=BUFFER_SIZE)
# datasets['test'] = test_data

# # Create the model
# BATCH_SIZE = 100
# output_size = 10
# depth = 2
# width = 50
# epochs = 5

# model_structure = {
#     'Input': tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     'Hidden': [(depth, width, 'relu')],
#     'Output': (output_size, 'softmax')
# }

# model = deep_learning.deep_model(datasets=datasets, model_structure=model_structure, batch_size=BATCH_SIZE, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], epochs=epochs)
