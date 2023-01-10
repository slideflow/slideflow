"""Functions for applying saved segmentation masks."""

import numpy as np
import slideflow as sf
from os.path import dirname, join, exists
from typing import Optional

class Segment:

    def __init__(self, source: Optional[str] = None) -> None:
        """QC function which loads a saved numpy mask.

        Args:
            source (str, optional): Path to search for qc mask.
                If None, will search in the same directory as the slide.
                Defaults to None.
        """
        self.source = source

    def __repr__(self):
        return "Segment(source={!r})".format(
            self.source
        )

    def __call__(self, wsi: "sf.WSI") -> None:
        """Applies a segmentation mask to a given slide from a saved npz file.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            None
        """
        from slideflow.slide.seg import Segmentation
        source = self.source if self.source is not None else dirname(wsi.path)
        if exists(join(source, wsi.name+'-masks.zip')):
            seg = Segmentation.load(join(source, wsi.name+'-masks.zip'))
            wsi.apply_segmentation(seg)
        return None