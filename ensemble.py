import os
import click
import slideflow as sf
import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------
@click.command()
@click.option('--project', '-p', required=True, help='Slideflow project', metavar='PATH')
@click.option('--outcome', '-o', required=True, help='Outcome label for training', metavar=str)
@click.option('--model', '-m', help='Reference model; will train on training slides used for this model.', metavar='PATH')
@click.option('--train', '-t', is_flag=True, default=False, help='Train the adversarial model.', metavar=bool)
@click.option('--validate', '-v', is_flag=True, default=False, help='Validate the adversarial model.', metavar=bool)
# @click.option('--annotations', '-a', help='Annotation file', metavar='PATH')
def main(
    project,
    outcome,
    model,
    train,
    validate
    # annotations
) -> None:
    """
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
        # Epochs naming is throwing an error ---  tensorflow.py 1856

    if validate:
        # Validate the adversarial model
        if model:
            print("Load Models")
            paths = [
            "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00019-histological_type-HP0/histological_type-HP0_adversarial_epoch",
            "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00020-histological_type-HP0/histological_type-HP0_adversarial_epoch",
            "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00021-histological_type-HP0/histological_type-HP0_adversarial_epoch",
            "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00022-histological_type-HP0/histological_type-HP0_adversarial_epoch",
            "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00023-histological_type-HP0/histological_type-HP0_adversarial_epoch"]

            adv_models = [tf.keras.models.load_model(path) for path in paths]

            print("Get Validation slides and prepare dataset")
            val_slides = sf.util.get_slides_from_model_manifest(model, 'validation')

            df = pd.read_csv("/home/prajval/DATA/PROJECTS/TCGA_BRCA/annotations.csv")
            slide = df["slide"]
            ht = df[outcome]
            lst = [list(a) for a in zip(ht, slide)]
            final = [] 
            for i in lst:
                if i[0] not in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']:
                    final.append(i)
            ood_slides = []
            for i in final:
                ood_slides.append(i[1])
            val_slides.extend(ood_slides)
            
            # prepare dataset
            dataset = P.dataset(tile_px=299, tile_um = 302, filters={'slide': val_slides})
            tf_dts = dataset.tensorflow(
                labels=dataset.labels('histological_type')[0],
                batch_size=64,
                infinite=False,
                normalizer=sf.util.get_model_normalizer(paths[1]),
                standardize=True,
                incl_slidenames=True)
            print(tf_dts)
                
            print("Starting ensemble")
            pred = []
            unc = []
            slide_label = []

            for batch_images, batch_labels, batch_slide in tqdm(tf_dts, total=dataset.num_tiles // 64):
        
                # pred_mean = adv_models[1](batch_images)
                # # print(type(pred_mean))
                # # print(pred_mean)
                # pred.append(pred_mean.numpy())
                # unc.append(pred_mean.numpy())
                
                batch_pred = [model(batch_images) for model in adv_models]

                pred_mean = tf.stack(batch_pred, axis = 0)      # dimension (5,64,2)
                pred.append(tf.reduce_mean(pred_mean, axis=0).numpy().tolist()) # dimension (64,2)

                pred_std = tf.stack(batch_pred, axis = 0)  
                unc.append(tf.math.reduce_std(pred_std, axis=0).numpy().tolist())
                
                b_slide = [i.decode("utf-8") for i in batch_slide.numpy()]
                slide_label.append(b_slide)

                # i += 1
                # if i > 3:
                #     break
                    
            # --- Save predictions in CSV format --------------------------------------
            print("csv conversion")
            slide_label = np.array(slide_label, dtype=object)
            slide_label = np.concatenate(slide_label, axis=0)
            df_slide = pd.DataFrame(data = slide_label,
                            columns = ["slide"])

            pred = np.array(pred, dtype=object)
            pred = np.concatenate(pred, axis=0)
            df_pred = pd.DataFrame(data = pred,
                            columns = ["cohort-y_pred0", "cohort-y_pred1"])

            unc = np.array(unc, dtype=object)
            unc = np.concatenate(unc, axis=0)
            df_unc = pd.DataFrame(data = unc,
                            columns = ["cohort-uncertainty0", "cohort-uncertainty1"])

            df = pd.concat([df_slide, df_pred, df_unc], axis=1)
            # print(df)
            df.to_csv("/home/prajval/DATA/PROJECTS/TCGA_LUNG/eval/ensemble_evaluations/4/tile_predictions_ensemble.csv")
        
            # for batch_images, batch_labels, batch_slide in tf_dts:
                
            #     batch_pred = [model(batch_images) for model in adv_models]

            #     pred_mean = tf.stack(batch_pred, axis = 0)      # dimension (5,64,2)
            #     pred.append(tf.reduce_mean(pred_mean, axis=0).numpy().tolist()) # dimension (64,2)

            #     pred_std = tf.stack(batch_pred, axis = 0)  
            #     unc.append(tf.math.reduce_std(pred_std, axis=0).numpy().tolist())
                
            #     slide_label.append(batch_slide)              
            
            # print("csv conversion")
            # slide_label = np.array(slide_label, dtype=object)
            # slide_label = np.concatenate(slide_label, axis=0)
            # df_slide = pd.DataFrame(data = slide_label,
            #               columns = ["slide"])

            # pred = np.array(pred, dtype=object)
            # pred = np.concatenate(pred, axis=0)
            # df_pred = pd.DataFrame(data = pred,
            #               columns = ["histological_type-y_pred0", "histological_type-y_pred1"])

            # unc = np.array(unc, dtype=object)
            # unc = np.concatenate(unc, axis=0)
            # df_unc = pd.DataFrame(data = unc,
            #               columns = ["histological_type-uncertainty0", "histological_type-uncertainty1"])

            # df = pd.concat([df_slide, df_pred, df_unc], axis=1)
            # print(df)

            # df.to_csv("/home/prajval/DATA/PROJECTS/TCGA_LUNG/eval/ensemble_evaluations/4/tile_predictions_ensemble.csv")
    
if __name__ == '__main__':
    mp.freeze_support()
    main()