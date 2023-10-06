import click
from os.path import dirname, realpath
from . import Studio

#----------------------------------------------------------------------------

@click.command()
@click.argument('slide', metavar='PATH', required=False)
@click.option('--model', '-m', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--project', '-p', help='Slideflow project.', metavar='PATH')
@click.option('--low_memory', '-l', is_flag=True, help='Low memory mode.', metavar=bool)
@click.option('--stylegan', '-g', is_flag=True, help='Enable StyleGAN support (requires PyTorch).', metavar=bool)
@click.option('--picam', '-pc', is_flag=True, help='Enable Picamera2 view (experimental).', metavar=bool)
@click.option('--camera', '-c', is_flag=True, help='Enable Camera (OpenCV) view (experimental).', metavar=bool)
@click.option('--cellpose', is_flag=True, help='Enable Cellpose segmentation (experimental).', metavar=bool)
def main(
    slide,
    model,
    project,
    low_memory,
    stylegan,
    picam,
    camera,
    cellpose
):
    """
    Whole-slide image viewer with deep learning model visualization tools.

    Optional PATH argument can be used specify which slide to initially load.
    """
    if low_memory is None:
        low_memory = False

    # Load widgets
    widgets = Studio.get_default_widgets()
    if stylegan:
        from .widgets.stylegan import StyleGANWidget
        widgets += [StyleGANWidget]

    if picam:
        from .widgets.picam import PicamWidget
        widgets += [PicamWidget]

    if camera:
        from .widgets.cvcam import CameraWidget
        widgets += [CameraWidget]

    if cellpose:
        from .widgets.segment import SegmentWidget
        widgets += [SegmentWidget]

    viz = Studio(low_memory=low_memory, widgets=widgets)
    viz.project_widget.search_dirs += [dirname(realpath(__file__))]

    # Load model.
    if model is not None:
        viz.load_model(model)

    if project is not None:
        viz.load_project(project)

    # Load slide(s).
    if slide:
        viz.load_slide(slide)

    # Run.
    viz.run()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
