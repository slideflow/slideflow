import os
import click
import slideflow as sf
import multiprocessing as mp

# ------------------------------------------------------------------------------------

@click.command()
@click.option('--project', '-p', required=True, help='Slideflow project', metavar='PATH')
@click.option('--outcome', '-o', required=True, help='Outcome label for training', metavar=str)
@click.option('--model', '-m', help='Reference model; will train on training slides used for this model.', metavar='PATH')
@click.option('--train', '-t', is_flag=True, default=False, help='Train the adversarial model.', metavar=bool)
def main(
    project,
    outcome,
    model,
    train
) -> None:
    r"""
    Train and generate uncertainty predictions using adversarial training,
    as implemented by Neural Structured Learning.

    Based on:

    - Tutorial (Google): https://www.tensorflow.org/neural_structured_learning/tutorials/adversarial_keras_cnn_mnist
    - GitHub (neural-structured-learning): https://github.com/tensorflow/neural-structured-learning/blob/master/g3doc/tutorials/adversarial_keras_cnn_mnist.ipynb

    # Objective

    Incorporate adversarial regularization into training to estimate of uncertainty

    # General schematic

    ## Training

    1. Train with adversarial regularization enabled, and without validation
            
        hp = sf.ModelParams(..., loss='negative_log_likelihood')
        P.train(..., params=hp, adversarial=True)
    """

    P = sf.Project(project)

    if train:
        # Train a model using the same train/validation split as the dropout-trained model,
        # for easier performance comparison
        if model:
            filter_kw = dict(
                filters={'slide': sf.util.get_slides_from_model_manifest(model, "training")}
            )
        else:
            filter_kw = {}
        hp = sf.ModelParams(
            tile_px=299,
            tile_um=302,
            batch_size=64,
            epochs=[1],
            dropout=0.2,
        )
        P.train(
            outcomes=outcome,
            pretrain=None,
            val_strategy=None,
            adversarial=True,
            params=hp,
            **filter_kw
        )
    
if __name__ == '__main__':
    mp.freeze_support()
    main()