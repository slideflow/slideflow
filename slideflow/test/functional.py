import importlib.util
import logging
import os
from os.path import exists, join
from typing import TYPE_CHECKING, List, Tuple, Union

import slideflow as sf
from PIL import Image
from slideflow.stats import SlideMap
from slideflow.test.utils import (handle_errors, test_multithread_throughput,
                                  test_throughput)
from rich import print
from tqdm import tqdm

spams_loader = importlib.util.find_spec('spams')

if TYPE_CHECKING:
    import multiprocessing


@handle_errors
def activations_tester(
    project: sf.Project,
    verbosity: int,
    passed: "multiprocessing.managers.ValueProxy",
    model: str,
    tile_px: int,
    **kwargs
) -> None:
    """Tests generation of intermediate layer activations.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)

    # Test activations generation.
    dataset = project.dataset(tile_px, 604)
    test_slide = dataset.slides()[0]

    df = project.generate_features(
        model=model,
        outcomes='category1',
        **kwargs
    )
    act_by_cat = df.activations_by_category(0).values()
    assert df.num_features == 1280  # mobilenet_v2
    assert df.num_classes == 2
    assert len(df.activations) == len(dataset.tfrecords())
    assert len(df.locations) == len(df.activations) == len(df.predictions)
    assert all([
        len(df.activations[s]) == len(df.predictions[s]) == len(df.locations[s])
        for s in df.activations
    ])
    assert len(df.activations_by_category(0)) == 2
    assert (sum([len(a) for a in act_by_cat])
            == sum([len(df.activations[s]) for s in df.slides]))
    lm = df.softmax_mean()
    l_perc = df.softmax_percent()
    l_pred = df.softmax_predict()
    assert len(lm) == len(df.activations)
    assert len(lm[test_slide]) == df.num_classes
    assert len(l_perc) == len(df.activations)
    assert len(l_perc[test_slide]) == df.num_classes
    assert len(l_pred) == len(df.activations)

    umap = SlideMap.from_features(df)
    if not exists(join(project.root, 'stats')):
        os.makedirs(join(project.root, 'stats'))
    umap.save_plot(join(project.root, 'stats', '2d_umap.png'))
    tile_stats, pt_stats, cat_stats = df.stats()
    top_features_by_tile = sorted(
        range(df.num_features),
        key=lambda f: tile_stats[f]['p']
    )
    for feature in top_features_by_tile[:5]:
        umap.save_3d(
            join(project.root, 'stats', f'3d_feature{feature}.png'),
            feature=feature
        )
    df.box_plots(
        top_features_by_tile[:5],
        join(project.root, 'box_plots')
    )

    # Test mosaic.
    mosaic = project.generate_mosaic(df)
    mosaic.save(join(project.root, "mosaic_test.png"), figsize=(15, 15))


@handle_errors
def feature_generator_tester(
    project: sf.Project,
    verbosity: int,
    passed: "multiprocessing.managers.ValueProxy",
    model: str,
) -> None:
    """Tests feature generation for MIL (and related) models.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    outdir = join(project.root, 'mil')
    project.generate_feature_bags(
        model,
        outdir=outdir,
        force_regenerate=True
    )


@handle_errors
def evaluation_tester(project, verbosity, passed, **kwargs) -> None:
    """Tests model evaluation.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    project.evaluate(**kwargs)


@handle_errors
def prediction_tester(project, verbosity, passed, **kwargs) -> None:
    """Tests model predictions.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    project.predict(**kwargs)


@handle_errors
def reader_tester(project, verbosity, passed, tile_px) -> None:
    """Tests TFRecord reading between backends and ensures identical results.

    Function must happen in an isolated process to free GPU memory when done.
    """
    dataset = project.dataset(tile_px, 604)
    tfrecords = dataset.tfrecords()
    batch_size = 128
    assert len(tfrecords)

    # Torch backend
    torch_results = []
    torch_dts = dataset.torch(
        labels=None,
        batch_size=batch_size,
        infinite=False,
        augment=False,
        standardize=False,
        num_workers=6,
        pin_memory=False
    )
    if verbosity < logging.WARNING:
        torch_dts = tqdm(
            torch_dts,
            leave=False,
            ncols=80,
            unit_scale=batch_size,
            total=dataset.num_tiles // batch_size
        )
    for images, labels in torch_dts:
        torch_results += [
            hash(str(img.numpy().transpose(1, 2, 0)))  # CWH -> WHC
            for img in images
        ]
    if verbosity < logging.WARNING:
        torch_dts.close()  # type: ignore
    torch_results = sorted(torch_results)

    # Tensorflow backend
    tf_results = []
    tf_dts = dataset.tensorflow(
        labels=None,
        batch_size=batch_size,
        infinite=False,
        augment=False,
        standardize=False
    )
    if verbosity < logging.WARNING:
        tf_dts = tqdm(
            tf_dts,
            leave=False,
            ncols=80,
            unit_scale=batch_size,
            total=dataset.num_tiles // batch_size
        )
    for images, labels in tf_dts:
        tf_results += [hash(str(img.numpy())) for img in images]
    if verbosity < logging.WARNING:
        tf_dts.close()
    tf_results = sorted(tf_results)

    assert len(torch_results) == len(tf_results) == dataset.num_tiles
    assert torch_results == tf_results


@handle_errors
def single_thread_normalizer_tester(
    project: sf.Project,
    verbosity: int,
    passed: "multiprocessing.managers.ValueProxy",
    methods: Union[List, Tuple],
    tile_px: int
) -> None:
    """Tests all normalization strategies and throughput.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    if not len(methods):
        methods = sf.norm.StainNormalizer.normalizers  # type: ignore
    dataset = project.dataset(tile_px, 604)
    v = f'[bold]({sf.backend()}-native)[/]'

    dts_kw = {'standardize': False, 'infinite': True}
    if sf.backend() == 'tensorflow':
        dts = dataset.tensorflow(None, None, **dts_kw)
        raw_img = next(iter(dts))[0].numpy()
    elif sf.backend() == 'torch':
        dts = dataset.torch(None, None, **dts_kw)
        raw_img = next(iter(dts))[0].permute(1, 2, 0).numpy()
    Image.fromarray(raw_img).save(join(project.root, 'raw_img.png'))
    for method in methods:
        if method in ('vahadane', 'vahadane_spams') and spams_loader is None:
            print("Skipping Vahadane (spams); SPAMS not installed.")
            continue
        gen_norm = sf.norm.StainNormalizer(method)
        vec_norm = sf.norm.autoselect(method)
        st_msg = '[yellow]SINGLE-thread[/]'

        # Save example image
        img = Image.fromarray(gen_norm.rgb_to_rgb(raw_img))
        img.save(join(project.root, f'{method}.png'))

        gen_tpt = test_throughput(dts, gen_norm)
        dur = f"[blue][{gen_tpt:.1f} img/s][/]"
        print(f"Testing {method} [{st_msg}]... DONE " + dur)
        if type(vec_norm) != type(gen_norm):
            # Save example image
            img = Image.fromarray(vec_norm.rgb_to_rgb(raw_img))
            img.save(join(project.root, f'{method}_vectorized.png'))

            vec_tpt = test_throughput(dts, vec_norm)
            dur = f"[blue][{vec_tpt:.1f} img/s][/]"
            print(f"Testing {method} {v} [{st_msg}]... DONE {dur}")


@handle_errors
def multi_thread_normalizer_tester(
    project: sf.Project,
    verbosity: int,
    passed: "multiprocessing.managers.ValueProxy",
    methods: Union[List, Tuple],
    tile_px: int
) -> None:
    """Tests all normalization strategies and throughput.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    if not len(methods):
        methods = sf.norm.StainNormalizer.normalizers  # type: ignore
    dataset = project.dataset(tile_px, 604)
    v = f'[bold]({sf.backend()}-native)[/]'

    for method in methods:
        if 'vahadane' in method:
            print("Skipping Vahadane throughput testing.")
            continue
        gen_norm = sf.norm.StainNormalizer(method)
        vec_norm = sf.norm.autoselect(method)
        mt_msg = '[magenta]MULTI-thread[/]'
        gen_tpt = test_multithread_throughput(dataset, gen_norm)
        dur = f"[blue][{gen_tpt:.1f} img/s][/]"
        print(f"Testing {method} [{mt_msg}]... DONE " + dur)
        if type(vec_norm) != type(gen_norm):
            vec_tpt = test_multithread_throughput(dataset, vec_norm)
            dur = f"[blue][{vec_tpt:.1f} img/s][/]"
            print(f"Testing {method} {v} [{mt_msg}]... DONE " + dur)


@handle_errors
def wsi_prediction_tester(
    project: sf.Project,
    verbosity: int,
    passed: "multiprocessing.managers.ValueProxy",
    model: str,
) -> None:
    """Tests predictions of whole-slide images.

    Function must happen in an isolated process to free GPU memory when done.
    """
    sf.setLoggingLevel(verbosity)
    dataset = project.dataset()
    slide_paths = dataset.slide_paths(source='TEST')
    patient_name = sf.util.path_to_name(slide_paths[0])
    project.predict_wsi(
        model,
        join(project.root, 'wsi'),
        filters={'patient': [patient_name]}
    )
