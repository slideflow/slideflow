"""Functions for saving/loading QC masks."""

import numpy as np
import slideflow as sf
from os.path import dirname, join, exists
from typing import Optional

class Save:

    def __init__(self, dest: Optional[str] = None) -> None:
        """QC function which saves the mask to a numpy file.

        When this QC method is applied to a slide, the current QC masks
        (e.g., as applied by the Otsu or Gaussian filtering methods) are saved
        to a numpy file. These saved masks can be loaded in the future
        using :class:`slideflow.slide.qc.Load`. Saving/loading masks saves time
        by allowing to avoid regenerating masks repeatedly.

        By default, masks are saved in the same folder as whole-slide images.

        .. code-block:: python

            from slideflow.slide import qc

            # Define a QC approach that auto-saves masks
            qc = [
                qc.Otsu(),
                qc.Save()
            ]
            P.extract_tiles(qc=qc)

            ...
            # Auto-load previously saved masks
            qc = [
                qc.Load()
            ]
            P.extract_tiles(qc=qc)

        Args:
            dest (str, optional): Path in which to save the qc mask.
                If None, will save in the same directory as the slide.
                Defaults to None.
        """
        self.dest = dest

    def __repr__(self):
        return "Save(dest={!r})".format(
            self.dest
        )

    def __call__(self, wsi: "sf.WSI") -> None:
        """Save a QC mask for a given slide as a numpy file.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            None
        """
        dest = self.dest if self.dest is not None else dirname(wsi.path)
        mask = wsi.qc_mask
        if mask:
            np.savez(join(dest, wsi.name+'_qc.npz'), mask=mask)
        return None


class Load:

    def __init__(self, source: Optional[str] = None) -> None:
        """QC function which loads a saved numpy mask.

        Loads and applies a QC mask which was saved by
        :class:`slideflow.slide.qc.Save`

        Args:
            source (str, optional): Path to search for qc mask.
                If None, will search in the same directory as the slide.
                Defaults to None.
        """
        self.source = source

    def __repr__(self):
        return "Load(source={!r})".format(
            self.source
        )

    def __call__(self, wsi: "sf.WSI") -> Optional[np.ndarray]:
        """Load a QC mask for a given slide from a numpy file.

        Args:
            wsi (sf.WSI): Whole-slide image.

        Returns:
            Optional[np.ndarray]: Returns the QC mask if a {slide}_qc.npz file
            was found, otherwise returns None.
        """
        source = self.source if self.source is not None else dirname(wsi.path)
        if exists(join(source, wsi.name+'_qc.npz')):
            return np.load(join(source, wsi.name+'_qc.npz'))['mask']
        else:
            return None