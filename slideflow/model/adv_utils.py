import tensorflow as tf
from slideflow import log

# -----------------------------------------------------------------------------

def convert_dataset(
    dataset: tf.data.Dataset,
    image_input_name: str = 'tile_image',
    label_input_name: str = 'label'
) -> tf.data.Dataset:
    """Convert a dataset into the format expected by the adversarial wrapper."""

    def convert_to_dictionaries(image, label):
        return {image_input_name: image["tile_image"], label_input_name: label}

    return dataset.map(convert_to_dictionaries)

def make_adversarial_model(model, loss, train_data):
    """Convert a model into a model with adversarial regularization.

    This step replaces model.compile().

    Example:

        train_data = convert_dataset(train_data)
        _model = adv_utils.make_adversarial_model(
            model, hp.get_loss()
        )
    """
    # Delayed imports
    try:
        import neural_structured_learning as nsl
    except ImportError:
        raise ImportError(
            'Adversarial training requires the package '
            '"neural_structured_learning", which could not be found.'
        )
    # Wrap the model with adversarial regularization,
    # using tensorflow/neural-structured-learning
    config = nsl.configs.make_adv_reg_config(
        multiplier=0.2,
        adv_step_size=0.05,
        adv_grad_norm='infinity'
    )
    adv_model = nsl.keras.AdversarialRegularization(model, adv_config=config)
    # Note: Cannot use hp.get_opt(), as this will result in a
    # LossScaleOptimizer wrapping error
    adv_model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['acc']
    )
    # Convert the dataset into the image/label format expected
    # by the adversarial wrapper
    train_data = convert_dataset(train_data)
    log.debug("Adversarial wrapping complete.")
    return adv_model, train_data