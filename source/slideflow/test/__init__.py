import tensorflow as tf
import slideflow as sf

from os.path import join

class TestSuite:
    '''Class to supervise standardized testing of slideflow pipeline.'''
    def __init__(self, test_project_dir):
        '''Initialize testing models.'''
        # Intiailize project
        self.SFP = sf.SlideflowProject(test_project_dir)

        # Initialize model
        models_dir = self.SFP.PROJECT['models_dir']
        tile_px = self.SFP.PROJECT['tile_px']
        outcomes = sf.util.get_outcomes_from_annotations(outcome, filters=filters, use_float=(model_type=='linear'))
        subfolder = sf.NO_LABEL if (not subfolder or subfolder=='') else subfolder
        train_tfrecords = glob(join(self.SFP.PROJECT['tfrecord_dir'], subfolder, "*.tfrecords"))
        SFM = sf.trainer.model.SlideflowModel(models_dir, tile_px, outcomes, train_tfrecords, None)

    def test_input_stream(self, outcome, balancing, batch_size=16, augment=True, subfolder=None, filters=None, model_type='categorical'):
        dataset, dataset_with_slidenames, num_tiles = SFM.build_dataset_inputs(SFM.TRAIN_TFRECORDS, batch_size=batch_size, balance=balancing, augment=augment, finite=False, include_slidenames=False)
        
    def test_all(self):
        '''Perform and report results of all available testing.'''
        self.test_input_stream()