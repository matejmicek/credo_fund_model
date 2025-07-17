import pandas as pd
import json
from enum import Enum
from typing import List, Tuple

class FundingStage(Enum):
    """Enum for funding stages, using values that appear at the start of funding round names."""
    SEED = "Seed Round"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    SERIES_D = "Series D"
    SERIES_E = "Series E"
    SERIES_F = "Series F" 