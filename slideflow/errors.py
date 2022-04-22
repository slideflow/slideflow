'''Slideflow module errors.'''


# --- CPLEX errors ------------------------------------------------------------
class CPLEXNotFoundError(Exception):
    def __init__(self):
        msg = 'CPLEX not found; unable to perform preserved-site validation.'
        super().__init__(msg)


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
        msg = 'Unable to find source {} in config {}'.format(source, config)
        super().__init__(msg)

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
        msg = 'Insufficient number of patients to generate validation dataset.'
        super().__init__(msg)


class MismatchedImageFormatsError(DatasetError):
    pass


# --- Mosaic & Heatmap Errors -------------------------------------------------
class HeatmapError(Exception):
    pass


class MosaicError(Exception):
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


# --- TFRecords errors --------------------------------------------------------
class TFRecordsError(Exception):
    pass


class EmptyTFRecordsError(Exception):
    pass


# --- Slide errors ------------------------------------------------------------
class SlideError(Exception):
    pass


class SlideNotFoundError(SlideError):
    pass


class QCError(SlideError):
    pass


# --- Stats errors ------------------------------------------------------------
class StatsError(Exception):
    pass


class SlideMapError(Exception):
    pass


# --- Backend errors ----------------------------------------------------------
class BackendError(Exception):
    pass


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
    def __init__(self):
        super().__init__('Model parameters file (params.json) not found.')
