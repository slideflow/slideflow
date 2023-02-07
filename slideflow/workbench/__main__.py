import click
from os.path import dirname, realpath
from . import Workbench

#----------------------------------------------------------------------------

@click.command()
@click.argument('slide', metavar='PATH', required=False)
@click.option('--model', '-m', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--project', '-p', help='Slideflow project.', metavar='PATH')
@click.option('--low_memory', '-l', is_flag=True, help='Low memory mode.', metavar=bool)
@click.option('--stylegan', '-g', is_flag=True, help='Enable StyleGAN support (requires PyTorch).', metavar=bool)
@click.option('--picam', '-c', is_flag=True, help='Enable Picamera2 view (experimental).', metavar=bool)
@click.option('--cellpose', is_flag=True, help='Enable Cellpose segmentation (experimental).', metavar=bool)
@click.option('--advanced', '-a', is_flag=True, help='Enable advanced StyleGAN options.', metavar=bool)
def main(
    slide,
    model,
    project,
    low_memory,
    stylegan,
    picam,
    cellpose,
    advanced
):
    """
    Whole-slide image viewer with deep learning model visualization tools.

    Optional PATH argument can be used specify which slide to initially load.
    """
    if low_memory is None:
        low_memory = False

    # Load widgets
    widgets = Workbench.get_default_widgets()
    if stylegan:
        from slideflow.workbench import stylegan_widgets
        from slideflow.workbench.seed_map_widget import SeedMapWidget
        from slideflow.gan.stylegan3.stylegan3.viz.renderer import Renderer as GANRenderer
        widgets += stylegan_widgets(advanced=advanced)
        widgets += [SeedMapWidget]

    if picam:
        from slideflow.workbench.picam_widget import PicamWidget
        widgets += [PicamWidget]

    if cellpose:
        from slideflow.workbench.segment_widget import SegmentWidget
        widgets += [SegmentWidget]

    viz = Workbench(low_memory=low_memory, widgets=widgets)
    viz.project_widget.search_dirs += [dirname(realpath(__file__))]

    # --- StyleGAN3 -----------------------------------------------------------
    if stylegan:
        viz.add_to_render_pipeline(GANRenderer(), name='stylegan')
        if advanced:
            viz._pane_w_div = 45
    # -------------------------------------------------------------------------

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
