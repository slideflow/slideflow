import csv
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import sys
import time
import traceback
from functools import wraps
from os.path import exists, join
from typing import Any, Callable, Dict, List, Optional

import requests
import slideflow as sf
from slideflow.util import ProgressBar
from slideflow.util import colors as col
from slideflow.util import log
from slideflow.util.spinner import Spinner


def process_isolate(func: Callable, project: sf.Project, **kwargs) -> bool:
    ctx = multiprocessing.get_context('spawn')
    passed = ctx.Manager().Value(bool, True)
    verbosity = sf.getLoggingLevel()
    process = ctx.Process(
        target=func,
        args=(project, verbosity, passed),
        kwargs=kwargs
    )
    process.start()
    process.join()
    return passed.value


def handle_errors(func):

    @wraps(func)
    def wrapper(project, verbosity, passed, **kwargs):
        try:
            func(project, verbosity, passed, **kwargs)
        except Exception as e:
            log.error(traceback.format_exc())
            passed.value = False

    return wrapper


def get_tcga_slides() -> Dict[str, str]:
    slides = [
        'TCGA-BJ-A2N9-01Z-00-DX1.CFCB1FA9-7890-4B1B-93AB-4066E160FBF5',
        'TCGA-BJ-A3PT-01Z-00-DX1.A307F39F-AE85-42F4-B705-11AF06F391D9',
        'TCGA-BJ-A45J-01Z-00-DX1.F3646444-749B-4583-A45D-17C580FCB866',
        'TCGA-DJ-A2PT-01Z-00-DX1.8C28F7F7-426A-4AAC-8AC6-D082F85C4D34',
        'TCGA-DJ-A4UQ-01Z-00-DX1.2F88113C-4F3B-4250-A7C3-5B01AB6ABE55',
        'TCGA-DJ-A13W-01Z-00-DX1.02059A44-7DF1-420D-BA48-587D611F34F5',
        'TCGA-DO-A1K0-01Z-00-DX1.5ED4011C-6AAA-4197-8044-1F69D55CEAEE',
        'TCGA-E3-A3E5-01Z-00-DX1.E7E8AB8B-695F-4158-A3C0-E2B801E07D2A',
        'TCGA-E8-A242-01Z-00-DX1.9DDBB5BB-696E-4C61-BF4A-464062403F04',
        'TCGA-EL-A3CO-01Z-00-DX1.7BF5F004-E7E6-4320-BA89-39D05657BBCB'
    ]
    uuids = [
        '0b0b560d-f3e7-4103-9b1b-d4981e00c0e7',
        '0eeb9df4-4cb0-4075-9e18-3861dea2ba05',
        '0c376805-5f09-4687-8e29-ad36b2171577',
        '1af4e340-38d3-4589-8a7b-6be3f207bc06',
        '0d0e4ddf-749c-44ba-aea9-989732e79d8d',
        '0c5592d5-b51c-406a-9dd5-72778e982f13',
        '0d78b583-ecf2-45f4-95a4-dc61057be898',
        '1a4242c5-495d-46f2-b87d-050acc6cef44',
        '1bcfd879-c48b-4232-b6a7-ff1337be9914',
        '0ac4f9a9-32f8-40b5-be0e-52ceeef7dbbf'
    ]
    return dict(zip(slides, uuids))


def download_from_tcga(uuid: str, dest: str, message: str = '') -> None:
    params = {'ids': [uuid]}
    data_endpt = "https://api.gdc.cancer.gov/data"
    response = requests.post(
        data_endpt,
        data=json.dumps(params),
        headers={"Content-Type": "application/json"},
        stream=True
    )
    response_head_cd = response.headers["Content-Disposition"]
    block_size = 4096
    file_size = int(response.headers.get('Content-Length', ''))
    pb = ProgressBar(file_size, leadtext=message)
    file_name = join(dest, re.findall("filename=(.+)", response_head_cd)[0])
    with open(file_name, "wb") as output_file:
        for chunk in response.iter_content(chunk_size=block_size):
            output_file.write(chunk)
            pb.increase_bar_value(block_size)
    pb.end()


def random_annotations(
    slides_path: Optional[str] = None
) -> List[List]:
    if slides_path:
        slides = [
            sf.util.path_to_name(f)
            for f in os.listdir(slides_path)
            if sf.util.path_to_ext(f).lower() in sf.util.SUPPORTED_FORMATS
        ][:10]
    else:
        slides = [f'slide{i}' for i in range(10)]
    annotations = [['patient', 'slide', 'dataset', 'category1', 'category2',
                    'linear1', 'linear2', 'time', 'event']]
    for s, slide in enumerate(slides):
        cat1 = ['A', 'B'][s % 2]
        cat2 = ['A', 'B'][s % 2]
        lin1 = random.random()
        lin2 = random.random()
        time = random.randint(0, 100)
        event = random.choice([0, 1])
        annotations += [[slide, slide, 'TEST', cat1,
                         cat2, lin1, lin2, time, event]]  # type: ignore
    return annotations


def _assert_valid_results(results):
    assert isinstance(results, dict)
    assert len(results)
    model = list(results.keys())[0]
    assert 'epochs' in results[model]
    assert isinstance (results[model]['epochs'], dict)
    assert len(results[model]['epochs'])


def test_throughput(
    dts: Any,
    normalizer: Optional[sf.norm.StainNormalizer] = None,
    s: int = 5,
) -> float:
    """Tests throughput of image normalization with a single thread.

    Returns:
        float: images/sec.
    """
    #from slideflow.norm.tensorflow import macenko
    #n= macenko.MacenkoNormalizer()
    start = -1  # type: float
    count = 0
    total_time = 0  # type: float
    for img, slide in dts:
        if sf.backend() == 'torch':
            if len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            else:
                img = img.permute(0, 2, 3, 1)

        #n.transform(img)
        if sf.backend() == 'tensorflow' and normalizer is not None:
            normalizer.tf_to_tf(img)
        if sf.backend() == 'torch' and normalizer is not None:
            normalizer.torch_to_torch(img)
        if start == -1:
            start = time.time()
        else:
            if len(img.shape) == 3:
                count += 1
            else:
                count += img.shape[0]
        if time.time() - start > s:
            total_time = count / (time.time() - start)
            break
    return total_time


def test_multithread_throughput(
    dataset: Any,
    normalizer: sf.norm.StainNormalizer,
    s: int = 5,
    batch_size: int = 32
) -> float:
    """Tests throughput of image normalization with multiple threads.

    Returns:
        float: images/sec.
    """
    if sf.backend() == 'tensorflow':
        dts = dataset.tensorflow(
            None,
            batch_size,
            standardize=False,
            infinite=True,
            normalizer=normalizer
        )
    elif sf.backend() == 'torch':
        dts = dataset.torch(
            None,
            batch_size,
            standardize=False,
            infinite=True,
            normalizer=normalizer,
        )
    return test_throughput(dts, s=s)


class TaskWrapper:
    '''Test wrapper to assist with logging.'''
    VERBOSITY = logging.DEBUG

    def __init__(self, message: str) -> None:
        self.message = message
        self.failed = False
        self.skipped = False
        self.start = time.time()
        if self.VERBOSITY >= logging.WARNING:
            self.spinner = Spinner(message)

    def __enter__(self):
        if self.VERBOSITY >= logging.WARNING:
            self.spinner.__enter__()
        else:
            print(self.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback) -> None:
        duration = time.time() - self.start
        if self.VERBOSITY >= logging.WARNING:
            self.spinner.__exit__(exc_type, exc_val, exc_traceback)
        exc_failed = (exc_type is not None
                      or exc_val is not None
                      or exc_traceback is not None)
        if self.failed or exc_failed:
            self._end_msg("FAIL", col.red, f' [{duration:.0f} s]')
        elif self.skipped:
            self._end_msg("SKIPPED", col.yellow, f' [{duration:.0f} s]')
        else:
            self._end_msg("DONE", col.green, f' [{duration:.0f} s]')
        if self.VERBOSITY < logging.WARNING:
            print()

    def _end_msg(
        self,
        end_str: str,
        color_func: Callable,
        trail: str,
        width: int = 80
    ) -> None:
        right_msg = f' {color_func(end_str)}{trail}'
        if len(self.message) > width:
            left_msg = self.message[:width]
        else:
            left_msg = self.message + " " * (width - len(self.message))
        sys.stdout.write(left_msg)
        sys.stdout.write('\b' * (len(end_str) + len(trail) + 1))
        sys.stdout.write(right_msg)
        sys.stdout.flush()
        print()

    def fail(self) -> None:
        self.failed = True

    def skip(self) -> None:
        self.skipped = True


class TestConfig:
    def __init__(
        self,
        path: str = '',
        slides: Optional[str] = None
    ) -> None:
        """Test Suite configuration.

        Args:
            path (str): Path to directory for test projects and data.
            slides (str): Specifies source of test slides. Either path to
                directory, or 'download' to download a set of slides for
                testing from TCGA.  If path to directory containing slides,
                will use subset of slides at random for testing.
        """
        random.seed(0)
        if path is None:
            path = ''
        if slides == 'download':
            slides_path = join(path, 'slides')
        elif slides is None:
            slides_path = path
        else:
            slides_path = slides
        if slides_path and not exists(slides_path):
            os.makedirs(slides_path)
        self.sources = {
            'TEST': {
                'slides': slides_path,
                'roi': join(path, 'roi'),
                'tiles': join(path, 'project', 'tiles', 'TEST'),
                'tfrecords': join(path, 'project', 'tfrecords', 'TEST')
            }
        }
        self.project_settings = {
            'name': 'TEST_PROJECT',
            'annotations': './annotations.csv',
            'dataset_config': join(path, 'datasets.json'),
            'sources': ['TEST'],
            'models_dir': './models',
            'eval_dir': './eval',
        }  # type: Dict[str, Any]
        if slides == 'download':
            tcga_slides = get_tcga_slides()
            with TaskWrapper("Downloading slides..."):
                supported = sf.util.SUPPORTED_FORMATS
                existing = [
                    sf.util.path_to_name(f)
                    for f in os.listdir(slides_path)
                    if sf.util.path_to_ext(f).lower() in supported
                ]
                for slide in [s for s in tcga_slides if s not in existing]:
                    download_from_tcga(
                        uuid=tcga_slides[slide],
                        dest=slides_path,
                        message=f"Downloading {col.green(slide)} from TCGA..."
                    )
        self.annotations = random_annotations(slides_path)
        assert len(self.annotations) > 1
        self.reference_model = None

    def create_project(
        self,
        path: str = 'test_project',
        overwrite: bool = False
    ) -> "sf.Project":
        """Creates a test project.

        Args:
            path (str): Directory at which to initialize the project.
            overwrite (bool, optional): Remove existing project files if
                present. Defaults to False.

        Returns:
            sf.Project: Test project.
        """
        if exists(join(path, 'settings.json')) and overwrite:
            shutil.rmtree(path)
        if exists(join(path, 'settings.json')):
            self.project = sf.Project(path)
        else:
            self.project = sf.Project(path, **self.project_settings)
        self.project.save()
        self.configure_sources()
        self.configure_annotations()
        return self.project

    def configure_sources(self) -> None:
        for source in self.sources.keys():
            self.project.add_source(
                source,
                slides=self.sources[source]['slides'],
                roi=self.sources[source]['roi'],
                tiles=self.sources[source]['tiles'],
                tfrecords=self.sources[source]['tfrecords'],
                path=self.project.dataset_config
            )

    def configure_annotations(self) -> None:
        outfile = self.project.annotations
        with open(outfile, 'w') as csv_outfile:
            csv_writer = csv.writer(csv_outfile, delimiter=',')
            for an in self.annotations:
                csv_writer.writerow(an)
        project_dataset = sf.Dataset(
            tile_px=299,
            tile_um=302,
            sources='TEST',
            config=self.project.dataset_config,
            annotations=self.project.annotations
        )
        loaded_slides = project_dataset.slides()
        for slide in [row[0] for row in self.annotations[1:]]:
            if slide not in loaded_slides:
                print()
                log.error(f"Failed to associate slide names ({slide}).")
                with open(outfile, 'r') as ann_read:
                    print(ann_read.read())
