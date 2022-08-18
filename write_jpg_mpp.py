import os
import click
import slideflow as sf
from PIL import Image

from slideflow.slide import TIF_EXIF_KEY_MPP

# ----------------------------------------------------------------------------

@click.command()
@click.option('--src', help='Source directory with JPEG images.', metavar='PATH')
@click.option('--dest', help='Directory in which to write modified images.', metavar='PATH')
@click.option('--mpp', help='Microns per pixel.', metavar=float, required=True)
def main(
    src,
    dest,
    mpp
):
    """Write a microns-per-pixel (MPP) value to EXIF data for JPEG images.

    Reads JPEG files from a source directory, writes the given microns-per-pixel
    value, and saves to a destination directory.
    """
    source_jpgs = [f for f in os.listdir(src) if sf.util.path_to_ext(f).lower() in ['jpeg', 'jpg']]
    if not len(source_jpgs):
        print("No source jpg/jpeg images found.")
    for src_jpg in source_jpgs:
        with Image.open(os.path.join(src, src_jpg)) as img:
            exif = img.getexif()
            if TIF_EXIF_KEY_MPP not in exif.keys():
                exif[TIF_EXIF_KEY_MPP] = mpp
                dest_jpg = os.path.join(dest, src_jpg)
                img.save(dest_jpg, exif=exif, quality=100)
                print(f"Wrote MPP={mpp} to {dest_jpg}")
            else:
                print(f"Skipping {src_jpg}; MPP already written")

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------