import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import slideflow as sf
from slideflow.util import log

if TYPE_CHECKING:
    import neptune
    from slideflow import Dataset


class NeptuneLog:
    '''Creates neptune runs and assists with run logging.'''

    def __init__(self, api_token: str, workspace: str) -> None:
        '''Initializes with a given Neptune API token and workspace name.'''

        self.api_token = api_token
        self.workspace = workspace

    def start_run(
        self,
        name: str,
        project: str,
        dataset: "Dataset",
        tags: Optional[List[str]] = None
    ) -> "neptune.Run":
        '''Starts a neptune run'''

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
            management.create_project(project_name, key=_id)

        self.run = neptune.init_run(project=project_name, api_token=self.api_token)
        run_loc = f'{self.workspace}/{project_name}'
        log.info(f'Neptune run {name} initialized at {run_loc}')
        self.run['sys/name'] = name
        for t in tags:
            self.run['sys/tags'].add(t)
        self.run['data/annotations_file'] = dataset.annotations_file
        return self.run

    def log_config(self, hp_data: Dict, stage: str) -> None:
        '''Logs model hyperparameter data according to the given stage
        ('train' or 'eval')'''

        from neptune.utils import stringify_unsupported

        proj_keys = ['dataset_config', 'sources']
        model_keys = ['model_name', 'model_type', 'k_fold_i', 'outcomes']
        if not hasattr(self, 'run'):
            raise ValueError(
                "Neptune run not yet initialized (start with start_run())"
            )
        for model_info_key in model_keys:
            self.run[model_info_key] = stringify_unsupported(hp_data[model_info_key])
        outcomes = {
            str(key): str(value)
            for key, value in hp_data['outcome_labels'].items()
        }
        validation_params = {
            key: hp_data[key]
            for key in hp_data.keys()
            if 'validation' in key
        }
        self.run['backend'] = sf.backend()
        self.run['project_info'] = {key: stringify_unsupported(hp_data[key]) for key in proj_keys}
        self.run['outcomes'] = outcomes
        self.run['model_params/validation'] = stringify_unsupported(validation_params)
        self._log_hp(hp_data, 'stage', 'stage')
        self._log_hp(hp_data, 'model_params/hp', 'hp')
        self._log_hp(hp_data, 'model_params/hp/pretrain', 'pretrain')
        self._log_hp(hp_data, 'model_params/resume_training', 'resume_training')
        self._log_hp(hp_data, 'model_params/checkpoint', 'checkpoint')
        self._log_hp(hp_data, 'model_params/filters', 'filters')
        if stage == 'train':
            self._log_hp(hp_data, 'input_features', 'input_features')
            self._log_hp(hp_data, 'input_feature_labels', 'input_feature_labels')
            self._log_hp(hp_data, 'model_params/max_tiles', 'max_tiles')
            self._log_hp(hp_data, 'model_params/min_tiles', 'min_tiles')
            self._log_hp(hp_data, 'full_model_name', 'full_model_name')
        else:
            self._log_hp(hp_data, 'eval/dataset', 'sources')
            self._log_hp(hp_data, 'eval/min_tiles', 'min_tiles')
            self._log_hp(hp_data, 'eval/max_tiles', 'max_tiles')

    def _log_hp(self, hp_data, run_key, hp_key) -> None:
        try:
            self.run[run_key] = hp_data[hp_key]
        except KeyError:
            log.debug(f"Unable to log Neptune hp_data key '{hp_key}'")


def list_log(run: "neptune.Run", label: str, val: Any, **kwargs: Any) -> None:
    # If only one value for a metric, log to .../[metric]
    # If more than one value for a metric (e.g. AUC for each category),
    # log to .../[metric]/[i]
    if isinstance(val, list):
        for idx, v in enumerate(val):
            run[f"{label}/{idx}"].log(v, **kwargs)
    else:
        run[label].log(val, **kwargs)
