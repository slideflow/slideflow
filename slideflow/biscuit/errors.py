class MatchError(Exception):
    pass


class ModelNotFoundError(MatchError):
    pass


class MultipleModelsFoundError(MatchError):
    pass


class EvalError(Exception):
    pass


class ThresholdError(Exception):
    pass


class ROCFailedError(Exception):
    pass


class PredsContainNaNError(Exception):
    pass