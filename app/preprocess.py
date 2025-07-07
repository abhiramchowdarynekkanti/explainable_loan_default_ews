
PRIORITY_FLAG_GROUPS = [
    {
        "flags": [
            "CODE_REJECT_REASON_XAP",
            "CODE_REJECT_REASON_HC",
            "CODE_REJECT_REASON_LIMIT",
            "CODE_REJECT_REASON_SCOFR"
        ],
        "default": "CODE_REJECT_REASON_LIMIT"
    },
    {
        "flags": [
            "NAME_CONTRACT_STATUS_Approved_x",
            "NAME_CONTRACT_STATUS_Completed_x",
            "NAME_CONTRACT_STATUS_Refused_x",
            "NAME_CONTRACT_STATUS_Signed_x",
            "NAME_CONTRACT_STATUS_Canceled_x"
        ],
        "default": "NAME_CONTRACT_STATUS_Completed_x"
    },
    {
        "flags": ["CREDIT_ACTIVE_Closed", "CREDIT_ACTIVE_Active"],
        "default": "CREDIT_ACTIVE_Closed"
    },
    {
        "flags": ["NAME_CLIENT_TYPE_Repeater", "NAME_CLIENT_TYPE_New"],
        "default": "NAME_CLIENT_TYPE_Repeater"
    },
    {
        "flags": [
            "NAME_CONTRACT_TYPE_Cash loans",
            "NAME_CONTRACT_TYPE_Consumer loans",
            "NAME_CONTRACT_TYPE_Revolving loans"
        ],
        "default": "NAME_CONTRACT_TYPE_Cash loans"
    },
    {
        "flags": [
            "NAME_PRODUCT_TYPE_walk-in",
            "NAME_PRODUCT_TYPE_x-sell",
            "NAME_PRODUCT_TYPE_XNA"
        ],
        "default": "NAME_PRODUCT_TYPE_walk-in"
    },
    {
        "flags": [
            "NAME_PORTFOLIO_Cash",
            "NAME_PORTFOLIO_POS",
            "NAME_PORTFOLIO_Cards",
            "NAME_PORTFOLIO_XNA"
        ],
        "default": "NAME_PORTFOLIO_Cash"
    },
    {
        "flags": [
            "PRODUCT_COMBINATION_Cash",
            "PRODUCT_COMBINATION_POS mobile with interest"
        ],
        "default": "PRODUCT_COMBINATION_Cash"
    },
    {
        "flags": [
            "NAME_PAYMENT_TYPE_Cash through the bank",
            "NAME_PAYMENT_TYPE_XNA"
        ],
        "default": "NAME_PAYMENT_TYPE_Cash through the bank"
    },
    {
        "flags": [
            "NAME_YIELD_GROUP_middle",
            "NAME_YIELD_GROUP_high",
            "NAME_YIELD_GROUP_low_normal",
            "NAME_YIELD_GROUP_XNA"
        ],
        "default": "NAME_YIELD_GROUP_middle"
    },
    {
        "flags": [
            "NAME_SELLER_INDUSTRY_Consumer electronics",
            "NAME_SELLER_INDUSTRY_Connectivity",
            "NAME_SELLER_INDUSTRY_XNA"
        ],
        "default": "NAME_SELLER_INDUSTRY_Consumer electronics"
    },
    {
        "flags": [
            "NAME_GOODS_CATEGORY_Consumer Electronics",
            "NAME_GOODS_CATEGORY_Mobile",
            "NAME_GOODS_CATEGORY_Audio/Video",
            "NAME_GOODS_CATEGORY_Computers",
            "NAME_GOODS_CATEGORY_XNA"
        ],
        "default": "NAME_GOODS_CATEGORY_Consumer Electronics"
    },
    {
        "flags": [
            "CHANNEL_TYPE_Credit and cash offices",
            "CHANNEL_TYPE_Country-wide",
            "CHANNEL_TYPE_AP+ (Cash loan)",
            "CHANNEL_TYPE_Stone",
            "CHANNEL_TYPE_Regional / Local"
        ],
        "default": "CHANNEL_TYPE_Credit and cash offices"
    },
    {
        "flags": [
            "WEEKDAY_APPR_PROCESS_START_MONDAY_y",
            "WEEKDAY_APPR_PROCESS_START_TUESDAY_y",
            "WEEKDAY_APPR_PROCESS_START_WEDNESDAY_y",
            "WEEKDAY_APPR_PROCESS_START_THURSDAY_y",
            "WEEKDAY_APPR_PROCESS_START_FRIDAY",
            "WEEKDAY_APPR_PROCESS_START_SATURDAY_y",
            "WEEKDAY_APPR_PROCESS_START_SUNDAY_y"
        ],
        "default": "WEEKDAY_APPR_PROCESS_START_MONDAY_y"
    },
    {
        "flags": [
            "NAME_TYPE_SUITE_Family_y",
            "NAME_TYPE_SUITE_Unaccompanied_y"
        ],
        "default": "NAME_TYPE_SUITE_Family_y"
    }
]
from app.constants import NUMERIC_DEFAULTS
from typing import Dict, List
import numpy as np
from app.encoder import load_features
# (paste the big feature_is_negative dict here, or import it)
from app.constants import  feature_is_negative
FEATURE_ORDER = load_features()

from app.constants import FLAG_COLUMNS

YES_VALUES = {"yes", "y", "true", "1", 1, True,"True","TRUE","Yes"}   # anything outside → 0

from app.constants import NUMERIC_DEFAULTS

"""def impute_numeric_defaults(sample: dict) -> dict:
    
    fixed = sample.copy()
    for feat, default_val in NUMERIC_DEFAULTS.items():
        if feat not in fixed or fixed[feat] in ("", None):
            fixed[feat] = default_val
        # Handle explicit NaN (e.g., from JS)                          ↓
        elif isinstance(fixed[feat], str):
            try:
                fixed[feat] = float(fixed[feat])
            except ValueError:
                fixed[feat] = default_val
    return fixed


"""


def apply_flag_priorities(sample: dict) -> dict:
    """
    For each mutually exclusive group, set only one to 1 (others to 0).
    If none provided, set default.
    """
    fixed = sample.copy()
    for group in PRIORITY_FLAG_GROUPS:
        found = False
        for flag in group["flags"]:
            if fixed.get(flag, 0) in [1, "1", "yes", "Yes", "true", True]:
                found = True
                fixed[flag] = 1.0
            else:
                fixed[flag] = 0.0
        # If none set, apply default
        if not found:
            for flag in group["flags"]:
                fixed[flag] = 1.0 if flag == group["default"] else 0.0
    return fixed



def encode_flag_columns(sample: dict) -> dict:
    """
    Convert flag columns (yes/no, true/false, 1/0) → float 1.0 / 0.0.
    Unknown or missing keys default to 0.0.
    """
    fixed = sample.copy()
    for col in FLAG_COLUMNS:
        raw = fixed.get(col, 0)          # missing → 0
        if isinstance(raw, str):
            fixed[col] = 1.0 if raw.strip().lower() in YES_VALUES else 0.0
        elif raw in YES_VALUES:
            fixed[col] = 1.0
        else:
            fixed[col] = 0.0
    return fixed

def ensure_negative_sign(sample: Dict[str, float]) -> Dict[str, float]:
    """
    Flip sign for any feature that must be negative but was sent as positive.
    """
    fixed = sample.copy()
    for feat, must_be_neg in feature_is_negative.items():
        if must_be_neg:
            val = fixed.get(feat, 0.0)
            if val > 0:
                fixed[feat] = -abs(val)
    return fixed
def impute_numeric_defaults(sample: dict) -> dict:
    """
    Fill missing / NaN / empty numeric fields with predefined medians/means.
    Now with debug logging for missing/imputed fields.
    """
    fixed = sample.copy()
    for feat, default_val in NUMERIC_DEFAULTS.items():
        val = fixed.get(feat, None)
        if val in ("", None):
            print(f"🟠 Imputing missing field: {feat} → {default_val}")
            fixed[feat] = default_val
        elif isinstance(val, str):
            try:
                fixed[feat] = float(val)
            except ValueError:
                print(f"🛑 Failed to parse {feat} = {val} → using default: {default_val}")
                fixed[feat] = default_val
    return fixed

def fix_numeric_zeros_with_defaults(sample: dict) -> dict:
    """
    Replaces zero-valued numeric fields with the corresponding values from NUMERIC_DEFAULTS.
    Assumes 0.0 means missing only for keys defined in NUMERIC_DEFAULTS.
    """
    fixed = sample.copy()
    for feat, default_val in NUMERIC_DEFAULTS.items():
        if feat in fixed and fixed[feat] == 0.0:
            print(f"🔁 Zero detected for {feat} → replacing with default {default_val}")
            fixed[feat] = default_val
    return fixed

def get_ordered_values(payload: Dict[str, float],
                       feature_order: List[str] = FEATURE_ORDER,
                       default: float = 0.0) -> List[float]:
    print("\n👣 [Stage 0] Raw payload from frontend:")
    print(payload)

    payload = encode_flag_columns(payload)

    payload = apply_flag_priorities(payload)

    payload = impute_numeric_defaults(payload)

    payload = fix_numeric_zeros_with_defaults(payload)
    print("\n✅ [Stage] After fixing zero-valued numerics:")
    print(payload)

    payload = ensure_negative_sign(payload)

    # Check for missing features in payload that exist in feature_order
    missing_feats = [f for f in feature_order if f not in payload]
    if missing_feats:
        print(f"\n🚨 Missing {len(missing_feats)} features in payload that are in feature_order:")
        for f in missing_feats:
            print(f"    • {f}")

    # Check for suspicious 0.0 values in key numeric fields
    print("\n🔍 Suspicious 0.0 values in key numeric fields:")
    for key in payload:
        if key.startswith("BUREAU_AMT_") or key.startswith("PREVAPP_AMT_"):
            if payload[key] == 0:
                print(f"    ⚠ {key} = 0.0")

    # Final ordered list
    ordered = [payload.get(f, default) for f in feature_order]

    return ordered

"""

def get_ordered_values(payload: Dict[str, float],
                       feature_order: List[str] = FEATURE_ORDER,
                       default: float = 0.0) -> List[float]:
    # 1. yes/no → 1/0
    payload = encode_flag_columns(payload)
    # 2. mutually‑exclusive default handling
    payload = apply_flag_priorities(payload)
    # 3. fill numeric medians/means
    payload = impute_numeric_defaults(payload)
    # 4. ensure negatives for day‑type fields
    payload = ensure_negative_sign(payload)
    # 5. reorder & pad
    return [payload.get(f, default) for f in feature_order]
"""

def preprocess_input(values: List[float], scaler):
    X = np.array(values, dtype=float).reshape(1, -1)
    transformed = scaler.transform(X)
    return scaler.transform(X)
