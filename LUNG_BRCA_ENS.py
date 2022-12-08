import os
os.environ['SF_BACKEND'] = 'tensorflow'
import multiprocessing
import slideflow as sf
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import neural_structured_learning as nsl

logging.getLogger('slideflow').setLevel(logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'


def main():
    P = sf.Project(root='/home/prajval/DATA/PROJECTS/TCGA_LUNG_BRCA')

    # for i in range(5):
    #     hp = sf.model.ModelParams(
    #             model='xception',
    #             tile_px=299,
    #             tile_um=302,
    #             batch_size=128,
    #             epochs=[1],
    #             early_stop=True,
    #             early_stop_method='accuracy',
    #             dropout=0.1,
    #             #uq=True,
    #             hidden_layer_width=1024,
    #             optimizer='Adam',
    #             learning_rate=0.0001,
    #             learning_rate_decay_steps=512,
    #             learning_rate_decay=0.98,
    #             loss='sparse_categorical_crossentropy',
    #             normalizer='reinhard_fast',
    #             include_top=False,
    #             hidden_layers=2,
    #             pooling='avg',
    #             augment='xyrjb'
    #         )

    #     P.train(
    #         outcomes='cohort',
    #         pretrain=None, # used for resnet50 and vgg19 
    #         val_k = 1,
    #         params=hp
    #     )

    print("Load Models")
    paths = [
    "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00001-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1",
    "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00002-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1",
    "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00003-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1",
    "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00004-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1",
    "/home/prajval/DATA/PROJECTS/TCGA_LUNG/models/00005-cohort-HP0-kfold1/cohort-HP0-kfold1_epoch1"]

    brca_slides_path = "/home/prajval/DATA/PROJECTS/TCGA_BRCA/models/00004-histological_type-HP0-kfold1/histological_type-HP0-kfold1_epoch3"

    adv_models = [tf.keras.models.load_model(path) for path in paths]

    print("Get Validation slides and prepare dataset")
    val_slides = sf.util.get_slides_from_model_manifest(paths[0], 'validation')
    print(len(val_slides))
    val_slides.extend(sf.util.get_slides_from_model_manifest(brca_slides_path, 'validation'))
    print(len(val_slides))
    
    # prepare dataset
    dataset = P.dataset(tile_px=299, tile_um = 302, filters={'slide': val_slides})
    # tf_dts = dataset.tensorflow(
    #     labels=dataset.labels('cohort')[0],
    #     batch_size=64,
    #     infinite=False,
    #     standardize=True,
    #     incl_slidenames=True)
    # print(tf_dts)
        
    # print("Starting ensemble")
    # pred = []
    # unc = []
    # slide_label = []
    # # i = 0

    # for batch_images, batch_labels, batch_slide in tf_dts:
        
    #     batch_pred = [model(batch_images) for model in adv_models]

    #     pred_mean = tf.stack(batch_pred, axis = 0)      # dimension (5,64,2)
    #     pred.append(tf.reduce_mean(pred_mean, axis=0).numpy().tolist()) # dimension (64,2)

    #     pred_std = tf.stack(batch_pred, axis = 0)  
    #     unc.append(tf.math.reduce_std(pred_std, axis=0).numpy().tolist())
        
    #     b_slide = [i.decode("utf-8") for i in batch_slide.numpy()]
    #     slide_label.append(b_slide)

    #     # i += 1
    #     # if i > 3:
    #     #     break
            
    
    # print("csv conversion")
    # slide_label = np.array(slide_label, dtype=object)
    # slide_label = np.concatenate(slide_label, axis=0)
    # df_slide = pd.DataFrame(data = slide_label,
    #                 columns = ["slide"])

    # pred = np.array(pred, dtype=object)
    # pred = np.concatenate(pred, axis=0)
    # df_pred = pd.DataFrame(data = pred,
    #                 columns = ["cohort-y_pred0", "cohort_pred1"])

    # unc = np.array(unc, dtype=object)
    # unc = np.concatenate(unc, axis=0)
    # df_unc = pd.DataFrame(data = unc,
    #                 columns = ["cohort-uncertainty0", "cohort-uncertainty1"])

    # df = pd.concat([df_slide, df_pred, df_unc], axis=1)
    # # print(df)
    # df.to_csv("/home/prajval/DATA/PROJECTS/TCGA_LUNG/eval/ensemble_evaluations/1/tile_predictions_ensemble.csv")

    print("Starting prediction")
    for path in paths:
        P.predict(path, dataset = dataset, batch_size = 64)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
