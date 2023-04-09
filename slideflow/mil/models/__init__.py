"""Multiple-instance learning (MIL) models"""

from .att_mil import Attention_MIL
from .clam import CLAM_MB, CLAM_SB
from .mil_fc import MIL_fc, MIL_fc_mc
from .transmil import TransMIL