from packaging import version
from google.protobuf import __version__ as protobuf_version

if version.parse(protobuf_version) < version.parse("3.21"):
    from ._proto3 import *
else:
    from ._proto4 import *

