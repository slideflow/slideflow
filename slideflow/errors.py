'''Slideflow module errors.'''

import os

# --- CPLEX / bonmin errors ---------------------------------------------------
class SolverNotFoundError(Exception):
    pass

# --- DatasetErrors -----------------------------------------------------------
class DatasetError(Exception):
    pass


class DatasetBalanceError(DatasetError):
    pass


class DatasetFilterError(DatasetError):
    pass


class DatasetClipError(DatasetError):
    pass


class SourceNotFoundError(DatasetError):
    def __init__(self, source, config):
        self.source = source
        self.config = config
        super().__init__(
            'Unable to find source {} in config {}'.format(source, config)
        )

    def __reduce__(self):
        return (SourceNotFoundError, (self.source, self.config))


class AnnotationsError(DatasetError):
    pass


class TFRecordsNotFoundError(DatasetError):
    def __init__(self):
        super().__init__('No TFRecords found.')


class DatasetSplitError(DatasetError):
    pass


class InsufficientDataForSplitError(DatasetError):
    def __init__(self):
        super().__init__(
            'Insufficient number of patients to generate validation dataset.'
        )


class MismatchedImageFormatsError(DatasetError):
    pass


class MismatchedSlideNamesError(DatasetError):
    pass


# --- Mosaic & Heatmap Errors -------------------------------------------------
class HeatmapError(Exception):
    pass


class MosaicError(Exception):
    pass

# --- TFRecord Heatmap Errors -------------------------------------------------

class CoordinateAlignmentError(Exception):
    pass

# --- Project errors ----------------------------------------------------------
class ProjectError(Exception):
    pass


# --- CLAM errors -------------------------------------------------------------
class CLAMError(Exception):
    pass


# --- Model and hyperparameters errors ----------------------------------------
class ModelError(Exception):
    pass


class ModelNotLoadedError(ModelError):
    def __init__(self):
        super().__init__('Model has not been loaded, unable to evaluate.')


class ModelParamsError(Exception):
    pass


class InvalidFeatureExtractor(Exception):
    pass

class UnrecognizedHyperparameterError(Exception):
    pass


# --- TFRecords errors --------------------------------------------------------
class TFRecordsError(Exception):
    pass


class TFRecordsIndexError(Exception):
    pass


class EmptyTFRecordsError(Exception):
    pass


class InvalidTFRecordIndex(Exception):
    pass


# --- Slide errors ------------------------------------------------------------
class SlideError(Exception):
    pass


class SlideLoadError(SlideError):
    pass


class SlideNotFoundError(SlideError):
    pass


class SlideMissingMPPError(SlideLoadError):
    pass

class IncompatibleBackendError(SlideLoadError):
    pass

class ROIError(SlideError):
    pass


class MissingROIError(ROIError):
    pass


class QCError(SlideError):
    pass


# --- Stats errors ------------------------------------------------------------
class StatsError(Exception):
    pass


class SlideMapError(Exception):
    pass


# --- Backend errors ----------------------------------------------------------
class UnrecognizedBackendError(Exception):
    def __init__(self):
        super().__init__(f"Unrecognized backend: {os.environ['SF_BACKEND']}")


# --- Features errors ---------------------------------------------------------
class FeaturesError(Exception):
    pass


# --- Normalizer errors -------------------------------------------------------
class NormalizerError(Exception):
    pass


class UserError(Exception):
    pass


class TileCorruptionError(Exception):
    '''Raised when image normalization fails due to tile corruption.'''
    pass


# --- Other errors ------------------------------------------------------------
class ModelParamsNotFoundError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            msg = 'Model parameters file (params.json) not found.'
        super().__init__(msg)


class SMACError(Exception):
    pass


class ChecksumError(Exception):
    pass


class AlignmentError(Exception):
    pass
