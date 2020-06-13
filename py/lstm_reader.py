import csv
import cv2
import os
from os import path
import random
from datetime import datetime

from keras import backend as K
from keras.layers import Activation, Bidirectional, Conv2D, Dense, LSTM, Input, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.preprocessing import sequence
import numpy as np
from pathlib import Path


CHARS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
         "-", ",", "'", "Æ", " "]
CHARS_WITH_NULL = CHARS + ["_"]
INPUT_W = 312  # 312 is the 95th percentile for the widths in training data
INPUT_H = 32


# Encode string to numbers
def encode_string(str):
    return [float(CHARS.index(c)) for c in str]


def decode_nums(nums):
    return [CHARS_WITH_NULL[int(num)] for num in nums]


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN  tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    '''
    y_true: tensor (samples, max_string_length) containing the truth labels.
    y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
    input_length: tensor (samples, 1) containing the sequence length for each batch item in y_pred.
    label_length: tensor (samples, 1) containing the sequence length for each batch item in y_true.
    '''
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# Build the recognition model. Strongly based on: https://keras.io/examples/image_ocr/
def buildModel(labels_max_len, chars_n):
    
    # Network parameters
    conv_filters = 16
    pool_size = 2
    time_dense_size = 32
    rnn_size = 168
    activation = "relu"

    # Convolutional feature extractor
    input_data = Input(name="input_layer", shape=(INPUT_W, INPUT_H, 1), dtype="float32")
    conv1 = Conv2D(conv_filters, kernel_size=(5, 5), padding="same", activation=activation, kernel_initializer="he_normal", name="conv1")(input_data)
    max1 = MaxPooling2D(pool_size=(pool_size, pool_size), name="max1")(conv1)
    conv2 = Conv2D(conv_filters, (3, 3), padding="same", activation=activation, kernel_initializer="he_normal", name="conv2")(max1)
    max2 = MaxPooling2D(pool_size=(pool_size, pool_size), name="max2")(conv2)

    # Cuts down input size going into RNN
    conv_to_rnn_dims = (INPUT_W // (pool_size ** 2), (INPUT_H // (pool_size ** 2)) * conv_filters)
    reshape = Reshape(target_shape=conv_to_rnn_dims, name="reshape")(max2)
    dense = Dense(time_dense_size, activation=activation, name="dense1")(reshape)

    # Recurrent layer
    rnn = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=0.1))(dense)

    # Transforms RNN output to character activations
    dense2 = Dense(chars_n, kernel_initializer="he_normal", name="dense2")(rnn)
    y_pred = Activation("softmax", name="softmax")(dense2)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name="the_labels", shape=[labels_max_len], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype='int64')
    label_length = Input(name="label_length", shape=[1], dtype='int64')

    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # The loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer="adam")

    return model


# Best path decoding
def best_path(prediction):
    maxIndices = []
    for arr in prediction:
        maxC = np.max(arr)
        maxI = np.where(arr == maxC)[0]
        maxIndices.append(int(maxI))

    # Collapse repeats
    previous = None
    collector = ""
    for maxI in maxIndices:
        current = CHARS_WITH_NULL[maxI]
        if current != previous and current != "_":
            collector = collector + current
        
        previous = current

    return collector


class LSTMClf:
    modelPath = "models/recognition_lstm_weights.h5"


    def __init__(self):
        trainingModel = buildModel(33, len(CHARS_WITH_NULL))
        trainingModel.load_weights(self.modelPath)
        imgInput = trainingModel.get_layer("input_layer").input
        output = trainingModel.get_layer("softmax").output
        self.model = Model(inputs=imgInput, outputs=output)
            

    # Predict the text in the input image. 
    # Image has to be a column major matrix, e.g. first axis is columns, second is rows
    def read_image(self, image):

        # Shorten images that are wider than 312px
        w, _h = image.shape
        if w > INPUT_W:
            image = cv2.resize(image, (INPUT_H, INPUT_W))

        image_scaled = (np.float16(image) / 255) - 0.5
        container = [image_scaled]
        container_pad = sequence.pad_sequences(container, maxlen=INPUT_W, value=255, dtype="float32", padding="post", truncating="post")
        container_pad = container_pad.reshape(1, INPUT_W, INPUT_H, 1)
        predictions = self.model.predict(container_pad)
        predStr = best_path(predictions[0]).replace("Æ", "Ae").strip()

        # Strip leading 1 character words, which are usually noise. Also takes care of single character full predictions.
        predSplit = predStr.split(" ")
        if len(predSplit) > 0 and len(predSplit[0]) == 1:
            predSplit = predSplit[1:len(predSplit)]
        
        return " ".join(predSplit)