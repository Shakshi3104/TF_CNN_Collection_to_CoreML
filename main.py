import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from coremotiontools import convert
from tensoract.applications.vgg16 import VGG16


if __name__ == '__main__':
    # load tf model
    model = VGG16(include_top=True)
    model_name = "VGG16"

    # convert to Core ML model
    class_labels = ["stay", "walk", "jog", "skip", "stUp", "stDown"]
    classifier_config = ct.ClassifierConfig(class_labels=class_labels)
    mlmodel = convert(model, classifier_config=classifier_config)

    mlmodel.short_description = "Activity Classifier ({})".format(model_name)
    mlmodel.input_description["input"] = "Input acceleration data to be classified"
    mlmodel.output_description["classLabel"] = "Most likely activity"
    mlmodel.output_description["Identity"] = "Probability of each activity"

    mlmodel.save("{}.mlmodel".format(model_name))

    # quantize model
    quantized_mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8, quantization_mode="linear")
    quantized_mlmodel.short_description = "Activity Classifier ({})".format(model_name + ", int 8bit")
    quantized_mlmodel.input_description["input"] = "Input acceleration data to be classified"
    quantized_mlmodel.output_description["classLabel"] = "Most likely activity"
    quantized_mlmodel.output_description["Identity"] = "Probability of each activity"

    quantized_mlmodel.save("{}Int8.mlmodel".format(model_name))
