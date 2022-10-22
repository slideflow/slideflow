import os
import click
import slideflow as sf
import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------
def _convert_dts_to_adv_dict( 
        dataset: tf.data.Dataset,
        image_input_name: str = 'tile_image',
        label_input_name: str = 'label'
    ) -> tf.data.Dataset:
        """Convert a dataset into the format expected by the adversarial wrapper."""
        print(dataset)
        def convert_to_dictionaries(image, label, slide):  
            return {image_input_name: image, label_input_name: label}

        return dataset.map(convert_to_dictionaries)

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
            print("Start 1")
            path = "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00019-histological_type-HP0/histological_type-HP0_adversarial_epoch"
            adv_model = tf.keras.models.load_model(path)

            print("Start 2")
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
                standardize=True,
                incl_slidenames=True)
            print(tf_dts)
                
            print("Start 3")
            pred = []
            unc = []
            slide_label = []
            i=0
            a = _convert_dts_to_adv_dict(tf_dts)
            batch_pred = adv_model.predict(a)
            # for batch_images, batch_labels, batch_slide in tf_dts:

            #     batch_images = tf.convert_to_tensor(batch_images)
            #     a = convert_to_dictionaries(batch_images, batch_labels)
            #     batch_pred = model.evaluate(a)

            #     pred_mean = tf.stack(batch_pred, axis = 0)      
            #     pred.append(tf.reduce_mean(pred_mean, axis=0).numpy().tolist()) 

            #     pred_std = tf.stack(batch_pred, axis = 0)  
            #     unc.append(tf.math.reduce_std(pred_std, axis=0).numpy().tolist())
                
            #     # slide_label.append(batch_slide) 
                
            #     i = i+1
            #     if i>3:
            #         break
            # print(pred, unc) #, slide_label)   
            print(batch_pred)  
            print(type(batch_pred))      

            with open("/home/prajval/DATA/PROJECTS/TCGA_BRCA/output.txt", "a") as f:
                print(batch_pred, file=f)     
                                
            
            print("Start 4")
            # slide_label = np.array(slide_label, dtype=object)
            # slide_label = np.concatenate(slide_label, axis=0)
            # df_slide = pd.DataFrame(data = slide_label,
            #               columns = ["slide"])

            pred = np.array(batch_pred)#, dtype=object)
            print(pred.shape)
            # pred = np.concatenate(pred, axis=0)
            df_pred = pd.DataFrame(data = pred,
                          columns = ["histological_type-y_pred0", "histological_type-y_pred1"])

            # unc = np.array(unc, dtype=object)
            # unc = np.concatenate(unc, axis=0)
            # df_unc = pd.DataFrame(data = unc,
            #               columns = ["histological_type-uncertainty0", "histological_type-uncertainty1"])

            # df = pd.concat([df_slide, df_pred, df_unc], axis=1)
            # print(df)

            df_pred.to_csv("/home/prajval/DATA/PROJECTS/TCGA_BRCA/ensemble_predictions_advrsarial_BRCA1.csv")       
    
if __name__ == '__main__':
    mp.freeze_support()
    main()