"""
Enumerations for risk levels and condition types.
"""
from enum import Enum


class RiskLevel(str, Enum):
    HIGH      = "HIGH"
    MEDIUM    = "MEDIUM"
    LOW       = "LOW"
    UNCERTAIN = "UNCERTAIN"


class ConditionType(str, Enum):
    BENIGN        = "BENIGN"
    MALIGNANT     = "MALIGNANT"
    PRE_MALIGNANT = "PRE_MALIGNANT"
    UNKNOWN       = "UNKNOWN"
