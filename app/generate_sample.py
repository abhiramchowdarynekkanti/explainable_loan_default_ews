import random
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Define your numeric feature ranges
# ----------------------------------------------------------

feature_ranges = {
    "AGE": (25, 60),
    "AMT_ANNUITY": (15000, 17000),
    "AMT_CREDIT": (420000, 480000),
    "AMT_GOODS_PRICE": (380000, 450000),
    "AMT_INCOME_TOTAL": (150000, 250000),
    "CNT_CHILDREN": (0, 3),
    "CNT_FAM_MEMBERS": (1, 5),
    "EXT_SOURCE_1": (0.05, 0.25),
    "EXT_SOURCE_2": (0.5, 0.85),
    "EXT_SOURCE_3": (0.4, 0.8),
    "HOUR_APPR_PROCESS_START": (8, 18),
    "OWN_CAR_AGE": (0, 20),
    "REGION_POPULATION_RELATIVE": (0.02, 0.05),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (0, 5),
    "APARTMENTS_AVG": (0, 1),
    "BASEMENTAREA_AVG": (0, 1),
    "COMMONAREA_AVG": (0, 1),
    "COMMONAREA_MEDI": (0, 1),
    "COMMONAREA_MODE": (0, 1),
    "DAYS_BIRTH": (-25000, -9000),
    "DAYS_EMPLOYED": (-30000, 0),
    "DAYS_ID_PUBLISH": (-8000, 0),
    "DAYS_LAST_PHONE_CHANGE": (-3000, 0),
    "DAYS_REGISTRATION": (-25000, 0),
    "DEF_30_CNT_SOCIAL_CIRCLE": (0, 20),
    "ENTRANCES_AVG": (0, 1),
    "FLOORSMAX_AVG": (0, 1),
    "FLOORSMIN_AVG": (0, 1),
    "LANDAREA_AVG": (0, 1),
    "LIVINGAPARTMENTS_AVG": (0, 1),
    "LIVINGAREA_AVG": (0, 1),
    "NONLIVINGAREA_AVG": (0, 1),
    "OBS_30_CNT_SOCIAL_CIRCLE": (0, 20),
    "OBS_60_CNT_SOCIAL_CIRCLE": (0, 20),
    "POS_CNT_INSTALMENT_FUTURE_MAX": (0, 50),
    "POS_CNT_INSTALMENT_FUTURE_MEAN": (0, 20),
    "POS_CNT_INSTALMENT_FUTURE_SUM": (0, 100),
    "POS_CNT_INSTALMENT_MAX": (0, 50),
    "POS_CNT_INSTALMENT_MEAN": (0, 20),
    "POS_CNT_INSTALMENT_SUM": (0, 100),
    "POS_MONTHS_BALANCE_MAX": (-120, 0),
    "POS_MONTHS_BALANCE_MEAN": (-60, 0),
    "POS_MONTHS_BALANCE_SUM": (-5000, 0),
    "POS_SK_DPD_DEF_MAX": (0, 10),
    "POS_SK_DPD_DEF_MEAN": (0, 5),
    "POS_SK_DPD_DEF_SUM": (0, 100),
    "POS_SK_DPD_MAX": (0, 10),
    "POS_SK_DPD_MEAN": (0, 5),
    "POS_SK_DPD_SUM": (0, 100),
    "POS_SK_ID_PREV_MAX": (1000000, 3000000),
    "POS_SK_ID_PREV_MEAN": (1000000, 3000000),
    "PREVAPP_AMT_ANNUITY_MAX": (0, 20000),
    "PREVAPP_AMT_ANNUITY_MEAN": (0, 20000),
    "PREVAPP_AMT_ANNUITY_SUM": (0, 100000),
    "PREVAPP_AMT_APPLICATION_MAX": (0, 500000),
    "PREVAPP_AMT_APPLICATION_MEAN": (0, 300000),
    "PREVAPP_AMT_APPLICATION_SUM": (0, 2000000),
    "PREVAPP_AMT_CREDIT_MAX": (0, 600000),
    "PREVAPP_AMT_CREDIT_MEAN": (0, 300000),
    "PREVAPP_AMT_CREDIT_SUM": (0, 2000000),
    "PREVAPP_AMT_DOWN_PAYMENT_MAX": (0, 100000),
    "PREVAPP_AMT_DOWN_PAYMENT_MEAN": (0, 50000),
    "PREVAPP_AMT_DOWN_PAYMENT_SUM": (0, 200000),
    "PREVAPP_AMT_GOODS_PRICE_MAX": (0, 600000),
    "PREVAPP_AMT_GOODS_PRICE_MEAN": (0, 400000),
    "PREVAPP_AMT_GOODS_PRICE_SUM": (0, 2000000),
    "PREVAPP_CNT_PAYMENT_MAX": (0, 60),
    "PREVAPP_CNT_PAYMENT_MEAN": (0, 30),
    "PREVAPP_DAYS_DECISION_MAX": (-3000, 0),
    "PREVAPP_DAYS_DECISION_MEAN": (-3000, 0),
    "PREVAPP_DAYS_FIRST_DRAWING_MEAN": (-3000, 0),
    "PREVAPP_DAYS_FIRST_DUE_MAX": (-3000, 0),
    "PREVAPP_DAYS_FIRST_DUE_MEAN": (-3000, 0),
    "PREVAPP_DAYS_LAST_DUE_1ST_VERSION_MAX": (-3000, 0),
    "PREVAPP_DAYS_LAST_DUE_1ST_VERSION_MEAN": (-3000, 0),
    "PREVAPP_DAYS_LAST_DUE_MAX": (-3000, 0),
    "PREVAPP_DAYS_LAST_DUE_MEAN": (-3000, 0),
    "PREVAPP_DAYS_TERMINATION_MAX": (-3000, 0),
    "PREVAPP_DAYS_TERMINATION_MEAN": (-3000, 0),
    "PREVAPP_HOUR_APPR_PROCESS_START_MAX": (0, 23),
    "PREVAPP_HOUR_APPR_PROCESS_START_MEAN": (0, 23),
    "PREVAPP_NFLAG_INSURED_ON_APPROVAL_MEAN": (0, 1),
    "PREVAPP_RATE_DOWN_PAYMENT_MAX": (0, 1),
    "PREVAPP_RATE_DOWN_PAYMENT_MEAN": (0, 1),
    "PREVAPP_SELLERPLACE_AREA_MAX": (0, 500),
    "PREVAPP_SELLERPLACE_AREA_MEAN": (0, 500),
    "PREVAPP_SK_ID_PREV_MAX": (1000000, 3000000),
    "PREVAPP_SK_ID_PREV_MEAN": (1000000, 3000000),
    "REGION_POPULATION_RELATIVE": (0.02, 0.05),
    "SK_ID_CURR": (100000, 700000),
    "TOTALAREA_MODE": (0, 1),
    "YEARS_BEGINEXPLUATATION_AVG": (0, 1),
    "YEARS_BUILD_AVG": (0, 1)
}

# ----------------------------------------------------------
# Define dummy columns
# ----------------------------------------------------------

dummy_columns = [
    "CHANNEL_TYPE_AP+ (Cash loan)",
    "CHANNEL_TYPE_Country-wide",
    "CHANNEL_TYPE_Credit and cash offices",
    "CHANNEL_TYPE_Regional / Local",
    "CHANNEL_TYPE_Stone",
    "NAME_CLIENT_TYPE_New",
    "NAME_CLIENT_TYPE_Repeater",
    "NAME_CONTRACT_TYPE_Cash loans",
    "NAME_CONTRACT_TYPE_Consumer loans",
    "NAME_CONTRACT_TYPE_Revolving loans",
    "NAME_PAYMENT_TYPE_Cash through the bank",
    "NAME_PAYMENT_TYPE_XNA",
    "NAME_PORTFOLIO_Cards",
    "NAME_PORTFOLIO_Cash",
    "NAME_PORTFOLIO_POS",
    "NAME_PORTFOLIO_XNA",
    "NAME_PRODUCT_TYPE_XNA",
    "NAME_PRODUCT_TYPE_walk-in",
    "NAME_PRODUCT_TYPE_x-sell",
    "NAME_SELLER_INDUSTRY_Connectivity",
    "NAME_SELLER_INDUSTRY_Consumer electronics",
    "NAME_SELLER_INDUSTRY_XNA",
    "NAME_TYPE_SUITE_Family_y",
    "NAME_TYPE_SUITE_Unaccompanied_y",
    "NAME_YIELD_GROUP_XNA",
    "NAME_YIELD_GROUP_high",
    "NAME_YIELD_GROUP_low_normal",
    "NAME_YIELD_GROUP_middle",
    "WEEKDAY_APPR_PROCESS_START_FRIDAY",
    "WEEKDAY_APPR_PROCESS_START_MONDAY_y",
    "WEEKDAY_APPR_PROCESS_START_SATURDAY_y",
    "WEEKDAY_APPR_PROCESS_START_SUNDAY_y",
    "WEEKDAY_APPR_PROCESS_START_THURSDAY_y",
    "WEEKDAY_APPR_PROCESS_START_TUESDAY_y",
    "WEEKDAY_APPR_PROCESS_START_WEDNESDAY_y",
    "CODE_REJECT_REASON_HC",
    "CODE_REJECT_REASON_LIMIT",
    "CODE_REJECT_REASON_SCOFR",
    "CODE_REJECT_REASON_XAP",
    "CREDIT_ACTIVE_Active",
    "CREDIT_ACTIVE_Closed",
    "CREDIT_TYPE_Consumer credit",
    "CREDIT_TYPE_Credit card",
    "NAME_CONTRACT_STATUS_Active_x",
    "NAME_CONTRACT_STATUS_Approved_x",
    "NAME_CONTRACT_STATUS_Canceled_x",
    "NAME_CONTRACT_STATUS_Completed_x",
    "NAME_CONTRACT_STATUS_Refused_x",
    "NAME_CONTRACT_STATUS_Signed_x",
    "NAME_GOODS_CATEGORY_Audio/Video",
    "NAME_GOODS_CATEGORY_Computers",
    "NAME_GOODS_CATEGORY_Consumer Electronics",
    "NAME_GOODS_CATEGORY_Mobile",
    "NAME_GOODS_CATEGORY_XNA",
    "PRODUCT_COMBINATION_Cash",
    "PRODUCT_COMBINATION_POS mobile with interest"
]

# ----------------------------------------------------------
# All feature names
# ----------------------------------------------------------

# Combine all known features
all_features = list(set(list(feature_ranges.keys()) + dummy_columns))

# ----------------------------------------------------------
# Generate samples
# ----------------------------------------------------------

samples = []

for sample_num in range(100):
    # Generate a base sample
    base_sample = {}

    for feat in all_features:
        if feat in dummy_columns:
            base_sample[feat] = random.randint(0, 1)
        elif feat in feature_ranges:
            low, high = feature_ranges[feat]
            val = random.uniform(low, high)
            base_sample[feat] = round(val, 4) if high - low < 10 else int(val)
        else:
            base_sample[feat] = round(random.uniform(0, 1), 4)

    # Now generate 10 variations of this base sample
    for var in range(10):
        new_sample = base_sample.copy()

        # Slightly perturb numeric fields
        for feat in feature_ranges.keys():
            if feat in new_sample:
                perturb = (random.uniform(-0.05, 0.05)) * (feature_ranges[feat][1] - feature_ranges[feat][0])
                new_val = new_sample[feat] + perturb
                low, high = feature_ranges[feat]
                # Clip to valid range
                new_val = max(low, min(high, new_val))
                if high - low < 10:
                    new_sample[feat] = round(new_val, 4)
                else:
                    new_sample[feat] = int(new_val)

        samples.append(new_sample)

# Convert to DataFrame and save
df = pd.DataFrame(samples)
df.to_csv("synthetic_data.csv", index=False)

print("Generated file: synthetic_data.csv with", df.shape[0], "rows and", df.shape[1], "columns.")
