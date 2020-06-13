from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def buildModel():
    num_filters = 10
    filter_size = 5
    activation = "elu"
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, kernel_size=(filter_size, filter_size), input_shape=(24, 24, 3), activation=activation),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(num_filters * 2, kernel_size=(3, 3), activation=activation),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(num_filters * 4, kernel_size=(3, 3), activation=activation),
        Dropout(0.2),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ])
    return model


# A class that initializes a convolutional neural network that can be used to classify images
# of connected components.
class CCConvClf:
    weightsPath = "models/component_cnn_clf_weights.h5"

    def __init__(self):
        print("--- Initializing connected component classifier ---")
        self.model = buildModel()
        self.model.load_weights(self.weightsPath)
        
    # Predict if the input image is a connected component that's a part of text or not.
    # Returns values in range [0, 1]
    def predict(self, img):
        return self.model.predict(img)