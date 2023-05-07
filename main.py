import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks, losses, layers, metrics, regularizers


# utility function that converts a pandas dataframe into a tensorflow dataset
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#create_an_input_pipeline_using_tfdata
def df_to_dataset(dataframe, shuffle=True):
    df = dataframe.copy()
    labels = df.pop('math score'), df.pop('reading score'), df.pop('writing score')
    df = {key: value[:, tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(10)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


#  utility function that returns a dataset with all of the features one-hot encoded
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#categorical_columns
def get_category_encoding_layer(name, dataset, max_tokens=None):
    index = layers.StringLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))


# Create a model with the chosen layers
def layer_model():
    layer = layers.concatenate(encoded_features)
    layer = layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.001))(layer)
    layer = layers.Dropout(0.5)(layer)
    layer = layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.001))(layer)
    return layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.001))(layer)


# Functions below this are for formatting and plotting the results
def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()


def format_output(data):
    m = np.array(data.pop('math score'))
    r = np.array(data.pop('reading score'))
    w = np.array(data.pop('writing score'))
    return m, r, w


# import csv file as a pandas dataframe
# https://www.kaggle.com/spscientist/students-performance-in-exams
dataframe = pd.read_csv("StudentsPerformance.csv")

# split the DataFrame into training, validation, and test sets using and 70/15/15 split
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#split_the_dataframe_into_training_validation_and_test_sets
train, val, test = np.split(dataframe.sample(frac=1), [int(0.7 * len(dataframe)), int(0.85 * len(dataframe))])

# convert the 3 datasets into dataframes
train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

# list of column names from the dataset
categorical_cols = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
all_inputs = []
encoded_features = []

# add the encoded categorical layers to the encoded_features list and adds the inputs to the all_inputs list
for header in categorical_cols:
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, max_tokens=10)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# three separate outputs for each score prediction
math_output = layers.Dense(1, name='Math', activation='relu')(layer_model())
reading_output = layers.Dense(1, name='Reading', activation='relu')(layer_model())
writing_output = layers.Dense(1, name='Writing', activation='relu')(layer_model())

# create a keras model with 1 input and 3 outputs
model = tf.keras.Model(all_inputs, [math_output, reading_output, writing_output])

#  compile the model and set parameters for optimization and loss
opt = tf.keras.optimizers.SGD(learning_rate=.001, momentum=0.05)
model.compile(optimizer=opt, loss=losses.mean_absolute_error, metrics=metrics.MeanAbsolutePercentageError())

# create model.png to visualize layers
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# fit the model and stop it early if starts to over train
history = model.fit(train_ds, epochs=100, validation_data=[val_ds],
                    callbacks=callbacks.EarlyStopping(monitor='loss', patience=5))

# create variables to evaluate the model and output the results
loss, math_loss, reading_loss, writing_loss, math_mape, reading_mape, writing_mape = model.evaluate(val_ds)
print(f'\nMath mean absolute percentage error= {math_mape}')
print(f'Reading mean absolute percentage error=: {reading_mape}')
print(f'Writing mean absolute percentage error=: {writing_mape}')

# make test predictions based on the test dataset
math_pred, reading_pred, writing_pred = model.predict(test_ds)

# plot the difference between the prediction results, then graph the losses
test_Y = format_output(test)
plot_diff(test_Y[0], math_pred, title='Math')
plot_diff(test_Y[1], reading_pred, title='Reading')
plot_diff(test_Y[2], writing_pred, title='Writing')
plot_metrics(metric_name='Math_loss', title='MATH LOSS', ylim=100)
plot_metrics(metric_name='Reading_loss', title='READING LOSS', ylim=100)
plot_metrics(metric_name='Writing_loss', title='WRITING LOSS', ylim=100)
