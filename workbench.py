import click
from slideflow.workbench import Workbench
from os.path import dirname, realpath

#----------------------------------------------------------------------------

@click.command()
@click.argument('slide', metavar='PATH', required=False)
@click.option('--model', '-m', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--project', '-p', help='Slideflow project.', metavar='PATH')
@click.option('--low_memory', '-l', is_flag=True, help='Low memory mode.', metavar=bool)
@click.option('--picam', '-c', is_flag=True, help='Enable Picamera2 view (experimental).', metavar=bool)
@click.option('--stylegan', '-g', is_flag=True, help='Enable StyleGAN viewer.', metavar=bool)
@click.option('--advanced', '-a', is_flag=True, help='Enable advanced StyleGAN options.', metavar=bool)
def main(
    slide,
    model,
    project,
    low_memory,
    picam,
    stylegan,
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
        widgets += stylegan_widgets(advanced=advanced)
    if picam:
        from slideflow.workbench.picam_widget import PicamWidget
        widgets += [PicamWidget]

    viz = Workbench(low_memory=low_memory, widgets=widgets)
    viz.project_widget.search_dirs += [dirname(realpath(__file__))]

    # --- StyleGAN3 -----------------------------------------------------------
    if stylegan:
        from slideflow.gan.stylegan3.stylegan3.viz.renderer import Renderer as GANRenderer
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
    while not viz.should_close():
        viz.draw_frame()
    viz.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
