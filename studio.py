import threading
import click
from os.path import dirname, realpath

__loaded__ = False

#----------------------------------------------------------------------------

def show_splash():
    """Show a splash screen while the GUI loads."""
    global __loaded__

    from tkinter import Tk, Canvas
    from PIL import ImageTk

    root     = Tk()
    img_file = "studio_logo_small_rect.png"
    image    = ImageTk.PhotoImage(file=img_file)
    w,h      = image.width(), image.height()

    root.withdraw()
    screen_width  = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width  / 2) - (w / 2)
    y = (screen_height / 2) - (h / 2)

    root.overrideredirect(True)
    root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')
    canvas = Canvas(root, highlightthickness=0)
    canvas.create_image(0, 0, image=image, anchor='nw')
    canvas.pack( expand=1, fill='both')

    # Render loop
    root.deiconify()
    while True:
        if __loaded__:
            root.destroy()
            break
        root.update_idletasks()
        root.update()

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
    global __loaded__

    # Start the splash screen
    threading.Thread(target=show_splash).start()

    from slideflow.studio import Studio

    if low_memory is None:
        low_memory = False

    # Load widgets
    widgets = Studio.get_default_widgets()
    if stylegan:
        from slideflow.studio import stylegan_widgets
        from slideflow.studio.seed_map_widget import SeedMapWidget
        from slideflow.gan.stylegan3.stylegan3.viz.renderer import Renderer as GANRenderer
        widgets += stylegan_widgets(advanced=advanced)
        widgets += [SeedMapWidget]

    if picam:
        from slideflow.studio.widgets.picam import PicamWidget
        widgets += [PicamWidget]

    if cellpose:
        from slideflow.studio.widgets.segment import SegmentWidget
        widgets += [SegmentWidget]

    # Experimental ROI annotation widget
    from slideflow.studio.widgets.annotation import AnnotationWidget
    widgets += [AnnotationWidget]

    viz = Studio(low_memory=low_memory, widgets=widgets)
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

    # Close the splash screen.
    __loaded__ = True

    # Run.
    viz.run()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
