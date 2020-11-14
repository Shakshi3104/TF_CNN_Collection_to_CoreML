import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model


# Add Reshape layer
def add_reshape_layer(model: Model):
    width = model.input.shape[1]
    channels = model.input.shape[2]

    # Add Reshape Layer
    # MLMultiArray is 1-dimensional NSNumber Array
    # input shape of keras is (window_size, )
    inputs = Input(shape=(width * channels, ), name="input")
    x = Reshape((width, channels), name="reshape")(inputs)
    
    # concat reshape_input and model
    model_output = model(x)

    reshaped_model = Model(inputs=inputs, outputs=model_output)
    return reshaped_model


# Convert tf.keras model to mlmodel
def convert(model: Model, model_name=None, nbits=32, quantization_mode="linear"):
    # add reshape layer
    model = add_reshape_layer(model)

    classifier_config = ct.ClassifierConfig(class_labels=["stay", "walk", "jog", "skip", "stUp", "stDown"])
    mlmodel = ct.convert(model, classifier_config=classifier_config)

    # Quantization options
    available_options = list(range(1, 9)) + [16]
    if nbits in available_options:
        mlmodel = quantization_utils.quantize_weights(mlmodel,
                                                      nbits=nbits,
                                                      quantization_mode=quantization_mode)

    # Add description
    if model_name is not None:
        # if model is quantized
        if nbits in available_options:
            if nbits == 16:
                model_name = "{}, float {} bit".format(model_name, nbits)
            else:
                model_name = "{}, int {} bit".format(model_name, nbits)

        mlmodel.short_description = "Activity Classifier ({})".format(model_name)
        mlmodel.input_description["input"] = "Input acceleration data to be classified"
        mlmodel.output_description["classLabel"] = "Most likely activity"
        mlmodel.output_description["Identity"] = "Probability of each activity"

    return mlmodel


if __name__ == "__main__":
    from tensoract.applications import vgg16
    from tensoract.applications import efficientnet

    models = {"VGG16": vgg16.VGG16,
              "EfficientNetB0": efficientnet.EfficientNetB0}

    for model_name, Model_build in models.items():
        model = Model_build(include_top=True,
                            weights="weights/{}/{}_hasc_weights_256.hdf5".format(model_name.lower(),
                                                                                 model_name.lower()))

        mlmodel = convert(model, model_name=model_name)
        mlmodel.save("mlmodels/{}.mlmodel".format(model_name))

        # quantized model
        mlmodel = convert(model, model_name=model_name, nbits=16)
        mlmodel.save("mlmodels/{}Float{}.mlmodel".format(model_name, 16))
