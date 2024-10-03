import csv
import logging
import multiprocessing
import os
import random
import shutil
import sys
import time
import traceback
from functools import wraps
from os.path import exists, join
from typing import Any, Callable, Dict, List, Optional
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn

import slideflow as sf
from slideflow.util import colors as col
from slideflow.util import log, ImgBatchSpeedColumn, download_from_tcga


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
        'b7d2f2de-bb30-425d-9bf3-4a621cdacb3e',
        '5d5da119-4a4b-4c2c-a071-7c230bbe15ea',
        'd2401115-b490-46c9-a679-a6a80adc7119',
        '284dbe84-5899-4f5e-a402-1aca8410b513',
        '024f49af-ada4-4682-80b1-7135eb33ebd2',
        'e5efae78-c235-4269-aa33-4b1e8dc0ac53',
        '951cd1b9-1b43-4118-91ef-6496722a74eb',
        'c0024a2a-ab58-4162-b4bc-787b08e23a74',
        'bca43d2b-4809-488c-af55-5c734939601c',
        'd1778916-4bdc-4c67-bb8f-63417212a62f',
    ]
    return dict(zip(slides, uuids))


def random_annotations(
    slides_path: Optional[str] = None
) -> List[List]:
    if slides_path:
        slides = [
            sf.util.path_to_name(f)
            for f in os.listdir(slides_path)
            if sf.util.is_slide(join(slides_path, f))
        ][:10]
    else:
        slides = [f'slide{i}' for i in range(10)]
    annotations = [['patient', 'slide', 'dataset', 'category1', 'category2',
                    'continuous1', 'continuous2', 'time', 'event']]
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
    start = -1  # type: float
    count = 0
    total_time = 0  # type: float

    pb = Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        ImgBatchSpeedColumn(),
        transient=True)
    task = pb.add_task("Testing...", total=None)  # type: ignore

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
            pb.start()
        else:
            if len(img.shape) == 3:
                count += 1
                pb.advance(task, 1)
            else:
                count += img.shape[0]
                pb.advance(task, img.shape[0])
        if time.time() - start > s:
            total_time = count / (time.time() - start)
            break

    pb.stop()
    return total_time


def test_multithread_throughput(
    dataset: Any,
    normalizer: sf.norm.StainNormalizer,
    s: int = 5,
    batch_size: int = 8
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

    def __enter__(self):
        if self.VERBOSITY < logging.WARNING:
            print(self.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback) -> None:
        duration = time.time() - self.start
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
                existing = [
                    sf.util.path_to_name(f)
                    for f in os.listdir(slides_path)
                    if sf.util.is_slide(join(slides_path, f))
                ]
                for slide in [s for s in tcga_slides if s not in existing]:
                    download_from_tcga(
                        uuid=tcga_slides[slide],
                        dest=slides_path,
                        message=f"Downloading [green]{slide}[/] from TCGA..."
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
        if sf.util.is_project(path) and overwrite:
            shutil.rmtree(path)
        if sf.util.is_project(path):
            self.project = sf.Project(path, create=True)
        else:
            self.project = sf.Project(path, **self.project_settings, create=True)
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
