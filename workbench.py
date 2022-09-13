import click
from slideflow.workbench import Workbench
from os.path import dirname, realpath

#----------------------------------------------------------------------------

@click.command()
@click.argument('slide', metavar='PATH', required=False)
@click.option('--browse-dir', help='Specify model path for the \'Browse...\' button', metavar='PATH')
@click.option('--model', help='Classifier network for categorical predictions.', metavar='PATH')
@click.option('--project', '-p', help='Slideflow project.', metavar='PATH')
@click.option('--low_memory', '-l', is_flag=True, help='Low memory mode.', metavar=bool)
@click.option('--picam', is_flag=True, help='Enable Picamera2.', metavar=bool)
@click.option('--activations', '-a', is_flag=True, help='Enable activations viewer.', metavar=bool)
@click.option('--stylegan', '-g', is_flag=True, help='Enable StyleGAN viewer.', metavar=bool)
def main(
    slide,
    browse_dir,
    model,
    project,
    low_memory,
    picam,
    activations,
    stylegan
):
    """Interactive model visualizer.

    Optional PATH argument can be used specify which .pkl file to load.
    """
    if low_memory is None:
        low_memory = False

    widgets = Workbench.get_default_widgets()

    if stylegan:
        from slideflow.gan.stylegan3.viz import latent_widget
        from slideflow.gan.stylegan3.viz import pickle_widget
        from slideflow.gan.stylegan3.viz import stylemix_widget
        from slideflow.gan.stylegan3.viz import trunc_noise_widget
        from slideflow.gan.stylegan3.viz import equivariance_widget

        widgets += [
            pickle_widget.PickleWidget,
            latent_widget.LatentWidget,
            stylemix_widget.StyleMixingWidget,
            trunc_noise_widget.TruncationNoiseWidget,
            equivariance_widget.EquivarianceWidget
        ]
    if picam:
        from slideflow.workbench.picam_widget import PicamWidget
        widgets += [PicamWidget]
    if activations:
        from slideflow.workbench.layer_widget import LayerWidget
        widgets += [LayerWidget]

    viz = Workbench(low_memory=low_memory, widgets=widgets)
    viz.project_widget.search_dirs += [dirname(realpath(__file__))]

    # --- StyleGAN3 -----------------------------------------------------------
    if stylegan:
        from slideflow.gan.stylegan3.viz.renderer import Renderer as GANRenderer
        renderer = GANRenderer(gan_px=512, gan_um=400)
        viz.add_to_render_pipeline(renderer)
    # -------------------------------------------------------------------------

    if browse_dir is not None:
        viz.slide_widget.search_dirs = [browse_dir]

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
