from slideflow.util import log

class NeptuneLog:
    '''Creates neptune runs and assists with run logging.'''

    def __init__(self, api_token, workspace):
        '''Initializes with a given Neptune API token and workspace name.'''

        self.api_token = api_token
        self.workspace = workspace

    def run(self, name, project, dataset, tags=[]):
        '''Starts a neptune run'''

        import neptune.new as neptune

        self.run = neptune.init(project=f'{self.workspace}/{project}', api_token=self.api_token)
        log.info(f'Neptune run {name} initialized at {self.workspace}/{project}')
        self.run['sys/name'] = name
        run_id = self.run['sys/id'].fetch()
        for t in tags:
            self.run['sys/name'].add(t)

        # Dataset info
        self.run['eval/dataset_filters'] = dataset.filters
        self.run['eval/annotations'] = dataset.annotations_file

        return self.run

    def log(self, hp_data, stage):
        '''Logs model hyperparameter data according to the given stage ('train' or 'eval')'''

        proj_info_keys = ['project_dir', 'dataset_config', 'datasets', 'annotations']
        model_info_keys = ['model_name',  'model_dir', 'model_type', 'k_fold_i', 'outcome_headers', 'neptune_run']

        self.run['stage'] = hp_data['stage']
        self.run['project_info'] = {key: hp_data[key] for key in proj_info_keys}
        self.run['model_info'] = {key: hp_data[key] for key in model_info_keys}
        self.run['model_info/outcome_labels'] = {str(key): str(value) for key, value in hp_data['outcome_labels'].items()}
        self.run['model_info/model_params/validation'] = {key: hp_data[key] for key in hp_data.keys() if 'validation' in key}
        self.run['model_info/model_params/hp'] = hp_data['hp']
        self.run['model_info/model_params/hp/pretrain'] = hp_data['pretrain']
        self.run['model_info/model_params/resume_training'] = hp_data['resume_training']
        self.run['model_info/model_params/checkpoint'] = hp_data['checkpoint']
        self.run['model_info/model_params/filters'] = hp_data['filters']

        if stage == 'train':
            self.run['model_info/slide_input_headers'] = hp_data['slide_input_headers']
            self.run['model_info/slide_input_labels'] = hp_data['slide_input_labels']
            self.run['model_info/model_params/max_tiles_per_slide'] = hp_data['max_tiles_per_slide']
            self.run['model_info/model_params/min_tiles_per_slide'] = hp_data['min_tiles_per_slide']
            self.run['model_info/full_model_name'] = hp_data['full_model_name']
        else:
            self.run['eval/project'] = hp_data['project_dir']
            self.run['eval/dataset'] = hp_data['datasets']
            self.run['eval/batch_size'] = hp_data['eval_batch_size']
            self.run['eval/min_tiles_per_slide'] = hp_data['min_tiles_per_slide']
            self.run['eval/max_tiles_per_slide'] = hp_data['max_tiles_per_slide']
