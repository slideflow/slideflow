import random
import slideflow as sf
from slideflow.util import log


class NeptuneLog:
    '''Creates neptune runs and assists with run logging.'''

    def __init__(self, api_token, workspace):
        '''Initializes with a given Neptune API token and workspace name.'''

        self.api_token = api_token
        self.workspace = workspace

    def start_run(self, name, project, dataset, tags=None):
        '''Starts a neptune run'''

        import neptune.new as neptune
        from neptune import management

        if tags is None:
            tags = []
        project_name = project.replace("_", "-").replace(".", "-")
        project_name = f'{self.workspace}/{project_name}'
        existing_projects = management.get_project_list()
        if project_name not in existing_projects:
            _id = f'SF{str(random.random())[2:9]}'
            log.debug(f"Neptune project {project_name} does not exist")
            log.info(f"Creating Neptune project {project_name} (ID: {_id})")
            management.create_project(project_name, _id)
        self.run = neptune.init(project=project_name, api_token=self.api_token)
        run_loc = f'{self.workspace}/{project_name}'
        log.info(f'Neptune run {name} initialized at {run_loc}')
        self.run['sys/name'] = name
        for t in tags:
            self.run['sys/tags'].add(t)
        self.run['data/annotations_file'] = dataset.annotations_file
        return self.run

    def log_config(self, hp_data, stage):
        '''Logs model hyperparameter data according to the given stage
        ('train' or 'eval')'''

        proj_keys = ['dataset_config', 'sources']
        model_keys = ['model_name', 'model_type', 'k_fold_i', 'outcomes']
        if not hasattr(self, 'run'):
            msg = "Neptune run not yet initialized (start with start_run())"
            raise ValueError(msg)
        for model_info_key in model_keys:
            self.run[model_info_key] = hp_data[model_info_key]
        outcomes = {
            str(key): str(value)
            for key, value in hp_data['outcome_labels'].items()
        }
        validation_params = {
            key: hp_data[key]
            for key in hp_data.keys()
            if 'validation' in key
        }
        self.run['stage'] = hp_data['stage']
        self.run['backend'] = sf.backend()
        self.run['project_info'] = {key: hp_data[key] for key in proj_keys}
        self.run['outcomes'] = outcomes
        self.run['model_params/validation'] = validation_params
        self.run['model_params/hp'] = hp_data['hp']
        self.run['model_params/hp/pretrain'] = hp_data['pretrain']
        self.run['model_params/resume_training'] = hp_data['resume_training']
        self.run['model_params/checkpoint'] = hp_data['checkpoint']
        self.run['model_params/filters'] = hp_data['filters']

        if stage == 'train':
            self.run['input_features'] = hp_data['input_features']
            self.run['input_feature_labels'] = hp_data['input_feature_labels']
            self.run['model_params/max_tiles'] = hp_data['max_tiles']
            self.run['model_params/min_tiles'] = hp_data['min_tiles']
            self.run['full_model_name'] = hp_data['full_model_name']
        else:
            self.run['eval/dataset'] = hp_data['sources']
            self.run['eval/min_tiles'] = hp_data['min_tiles']
            self.run['eval/max_tiles'] = hp_data['max_tiles']


def list_log(run, label, val, **kwargs):
    # If only one value for a metric, log to .../[metric]
    # If more than one value for a metric (e.g. AUC for each category),
    # log to .../[metric]/[i]
    if isinstance(val, list):
        for idx, v in enumerate(val):
            run[f"{label}/{idx}"].log(v, **kwargs)
    else:
        run[label].log(val, **kwargs)
