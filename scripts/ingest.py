from __future__ import annotations
from pathlib import Path
import polars as pl

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PATH_A = DATA_DIR / "Dataset_A_sample.csv"
PATH_B = DATA_DIR / "Dataset_B_sample.csv"
PATH_C = DATA_DIR / "Dataset_C.csv"

def normalize_text(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"[^A-Z0-9 ]+", "")
    )

def normalize_zip(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.extract(r"(\d{5})", 1)

def normalize_npi(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.extract(r"(\d{10})", 1)
# --------------------------
# Dataset A -> providers_a (1 row per NPI)
# --------------------------
a = (
    pl.scan_csv(PATH_A, infer_schema_length=5000, ignore_errors=True)
    .with_columns([
        normalize_npi("Rndrng_NPI").alias("npi"),
        normalize_text("Rndrng_Prvdr_First_Name").alias("first_name"),
        normalize_text("Rndrng_Prvdr_MI").alias("middle_name"),
        normalize_text("Rndrng_Prvdr_Last_Org_Name").alias("last_or_org_name"),
        normalize_text("Rndrng_Prvdr_Crdntls").alias("credentials"),
        normalize_text("Rndrng_Prvdr_St1").alias("street1"),
        normalize_text("Rndrng_Prvdr_St2").alias("street2"),
        normalize_text("Rndrng_Prvdr_City").alias("city"),
        normalize_text("Rndrng_Prvdr_State_Abrvtn").alias("state"),
        normalize_zip("Rndrng_Prvdr_Zip5").alias("zip5"),
        normalize_text("Rndrng_Prvdr_Type").alias("provider_type"),
        pl.col("HCPCS_Cd").cast(pl.Utf8).alias("hcpcs_cd"),
        pl.col("Tot_Benes").cast(pl.Float64).alias("tot_benes"),
        pl.col("Tot_Srvcs").cast(pl.Float64).alias("tot_srvcs"),
        pl.col("Avg_Mdcr_Stdzd_Amt").cast(pl.Float64).alias("avg_std_amt"),
    ])
    .select([
        "npi","first_name","middle_name","last_or_org_name","credentials",
        "street1","street2","city","state","zip5","provider_type",
        "hcpcs_cd","tot_benes","tot_srvcs","avg_std_amt"
    ])
)


providers_a = (
    a.filter(pl.col("npi").is_not_null())
     .group_by("npi")
     .agg([
        pl.first("first_name").alias("first_name"),
        pl.first("middle_name").alias("middle_name"),
        pl.first("last_or_org_name").alias("last_or_org_name"),
        pl.first("credentials").alias("credentials"),
        pl.first("street1").alias("street1"),
        pl.first("street2").alias("street2"),
        pl.first("city").alias("city"),
        pl.first("state").alias("state"),
        pl.first("zip5").alias("zip5"),
        pl.first("provider_type").alias("provider_type"),

        # evidence/aggregation features (useful later)
        pl.n_unique("hcpcs_cd").alias("n_unique_hcpcs"),
        pl.sum("tot_benes").alias("sum_benes"),
        pl.sum("tot_srvcs").alias("sum_srvcs"),
        pl.mean("avg_std_amt").alias("mean_avg_std_amt"),

        # drift indicators (EDA + later features)
        pl.n_unique("street1").alias("n_unique_street1"),
        pl.n_unique("city").alias("n_unique_city"),
        pl.n_unique("last_or_org_name").alias("n_unique_name"),
     ])
)
# --------------------------
# Dataset B -> providers_b (1 row per Profile_ID, NPI optional)
# --------------------------
b = (
    pl.scan_csv(PATH_B, infer_schema_length=5000, ignore_errors=True)
    .with_columns([
        pl.col("Covered_Recipient_Profile_ID").cast(pl.Utf8).alias("profile_id"),
        normalize_npi("Covered_Recipient_NPI").alias("npi"),
        normalize_text("Covered_Recipient_First_Name").alias("first_name"),
        normalize_text("Covered_Recipient_Middle_Name").alias("middle_name"),
        normalize_text("Covered_Recipient_Last_Name").alias("last_name"),
        normalize_text("Covered_Recipient_Name_Suffix").alias("suffix"),
        normalize_text("Recipient_Primary_Business_Street_Address_Line1").alias("street1"),
        normalize_text("Recipient_Primary_Business_Street_Address_Line2").alias("street2"),
        normalize_text("Recipient_City").alias("city"),
        normalize_text("Recipient_State").alias("state"),
        normalize_zip("Recipient_Zip_Code").alias("zip5"),
        normalize_text("Covered_Recipient_Specialty_1").alias("specialty_1"),
        normalize_text("Covered_Recipient_Primary_Type_1").alias("primary_type_1"),
        normalize_text("Covered_Recipient_License_State_code1").alias("license_state_1"),
        pl.col("Total_Amount_of_Payment_USDollars").cast(pl.Float64).alias("payment_amount"),
        pl.col("Program_Year").cast(pl.Int32).alias("program_year"),
    ])
    .select([
        "profile_id","npi","first_name","middle_name","last_name","suffix",
        "street1","street2","city","state","zip5",
        "specialty_1","primary_type_1","license_state_1",
        "payment_amount","program_year"
    ])
)

providers_b = (
    b.filter(pl.col("profile_id").is_not_null())
     .group_by("profile_id")
     .agg([
        pl.first("npi").alias("npi"),
        pl.first("first_name").alias("first_name"),
        pl.first("middle_name").alias("middle_name"),
        pl.first("last_name").alias("last_name"),
        pl.first("suffix").alias("suffix"),
        pl.first("street1").alias("street1"),
        pl.first("street2").alias("street2"),
        pl.first("city").alias("city"),
        pl.first("state").alias("state"),
        pl.first("zip5").alias("zip5"),
        pl.first("specialty_1").alias("specialty_1"),
        pl.first("primary_type_1").alias("primary_type_1"),
        pl.first("license_state_1").alias("license_state_1"),

        # payment evidence (useful later)
        pl.len().alias("n_payment_rows"),
        pl.sum("payment_amount").alias("sum_payment_amount"),
        pl.mean("payment_amount").alias("mean_payment_amount"),
        pl.n_unique("program_year").alias("n_years"),

        # drift indicators
        pl.n_unique("street1").alias("n_unique_street1"),
        pl.n_unique("city").alias("n_unique_city"),
        pl.n_unique("last_name").alias("n_unique_last_name"),
     ])
)
# --------------------------
# Dataset C -> providers_c (1 row per NPI)
# --------------------------
c = (
    pl.scan_csv(
        PATH_C,
        infer_schema_length=5000,
        ignore_errors=True,
        encoding="utf8-lossy",
    )
    .with_columns([
        normalize_npi("NPI").alias("npi"),
        normalize_text("FIRST_NAME").alias("first_name"),
        normalize_text("MDL_NAME").alias("middle_name"),
        normalize_text("LAST_NAME").alias("last_name"),
        normalize_text("ORG_NAME").alias("org_name"),
        normalize_text("STATE_CD").alias("state"),
        pl.col("PROVIDER_TYPE_CD").cast(pl.Utf8).alias("provider_type_cd"),
        normalize_text("PROVIDER_TYPE_DESC").alias("provider_type_desc"),
    ])
    .select([
        "npi",
        "first_name",
        "middle_name",
        "last_name",
        "org_name",
        "state",
        "provider_type_cd",
        "provider_type_desc",
    ])
)

providers_c = (
    c.filter(pl.col("npi").is_not_null())
     .group_by("npi")
     .agg([
        pl.first("first_name").alias("first_name"),
        pl.first("middle_name").alias("middle_name"),
        pl.first("last_name").alias("last_name"),
        pl.first("org_name").alias("org_name"),
        pl.first("state").alias("state"),
        pl.first("provider_type_cd").alias("provider_type_cd"),
        pl.first("provider_type_desc").alias("provider_type_desc"),
        pl.n_unique("provider_type_desc").alias("n_unique_provider_type_desc"),
     ])
)
# --------------------------
# Write outputs
# --------------------------
pa = providers_a.collect(engine="streaming")
pb = providers_b.collect(engine="streaming")
pc = providers_c.collect(engine="streaming")

pa.write_parquet(OUT_DIR / "providers_a.parquet")
pb.write_parquet(OUT_DIR / "providers_b.parquet")
pc.write_parquet(OUT_DIR / "providers_c.parquet")

print("Saved provider tables:")
print(" - outputs/providers_a.parquet:", pa.shape)
print(" - outputs/providers_b.parquet:", pb.shape)
print(" - outputs/providers_c.parquet:", pc.shape)
