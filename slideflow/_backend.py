"""Configure the deep learning and slide reading backends."""

import os
import importlib.util

# Deep learning backend - use Tensorflow if available.
_valid_backends = ('torch', 'tensorflow')
if 'SF_BACKEND' not in os.environ:
    if importlib.util.find_spec('torch'):
        os.environ['SF_BACKEND'] = 'torch'
    elif importlib.util.find_spec('tensorflow'):
        os.environ['SF_BACKEND'] = 'tensorflow'
    else:
        os.environ['SF_BACKEND'] = 'torch'
elif os.environ['SF_BACKEND'] not in _valid_backends:
    raise ValueError("Unrecognized backend set via environmental variable "
                    "SF_BACKEND: {}. Expected one of: {}".format(
                        os.environ['SF_BACKEND'],
                        ', '.join(_valid_backends)
                    ))

# Slide backend - use cuCIM if available.
_valid_slide_backends = ('cucim', 'libvips')
if 'SF_SLIDE_BACKEND' not in os.environ:
    os.environ['SF_SLIDE_BACKEND'] = 'libvips'
    if importlib.util.find_spec('cucim'):
        import cucim
        if cucim.is_available():
            os.environ['SF_SLIDE_BACKEND'] = 'cucim'
elif os.environ['SF_SLIDE_BACKEND'] not in _valid_slide_backends:
    raise ValueError("Unrecognized slide backend set via environmental variable"
                    " SF_SLIDE_BACKEND: {}. Expected one of: {}".format(
                        os.environ['SF_SLIDE_BACKEND'],
                        ', '.join(_valid_slide_backends)
                    ))

# -----------------------------------------------------------------------------

def backend():
    return os.environ['SF_BACKEND']


def slide_backend():
    return os.environ['SF_SLIDE_BACKEND']
