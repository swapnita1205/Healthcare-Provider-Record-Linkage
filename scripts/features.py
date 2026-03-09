from pathlib import Path
import polars as pl

from matching_utils import build_idf_from_texts, features_row as _features_row

OUT_DIR = Path("outputs")
CAND_DIR = OUT_DIR / "candidates"
FEAT_DIR = OUT_DIR / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# Load provider tables + candidates
pa = pl.read_parquet(OUT_DIR / "providers_a.parquet")
pb = pl.read_parquet(OUT_DIR / "providers_b.parquet")
pc = pl.read_parquet(OUT_DIR / "providers_c.parquet")

cand_ba = pl.read_parquet(CAND_DIR / "cand_ba.parquet")
cand_bc = pl.read_parquet(CAND_DIR / "cand_bc.parquet")

# Polars struct schema used as return_dtype in map_elements
FEATURE_STRUCT = pl.Struct([
    pl.Field("sim_jw_fullname", pl.Float64),
    pl.Field("sim_jw_lastname", pl.Float64),
    pl.Field("sim_lev_fullname", pl.Float64),
    pl.Field("sim_lev_lastname", pl.Float64),
    pl.Field("sim_soundex_last", pl.Int8),

    pl.Field("sim_jacc_fullname", pl.Float64),
    pl.Field("sim_jacc_lastname", pl.Float64),

    pl.Field("sim_tfidf_name", pl.Float64),
    pl.Field("sim_char3_name", pl.Float64),

    pl.Field("first_initial_match", pl.Int8),
    pl.Field("state_match", pl.Int8),
    pl.Field("zip_match", pl.Int8),
    pl.Field("zip3_match", pl.Int8),
    pl.Field("sim_jw_city", pl.Float64),
    pl.Field("sim_jw_street1", pl.Float64),

    pl.Field("org_keyword_match", pl.Int8),
    pl.Field("org_vs_person_conflict", pl.Int8),
    pl.Field("credential_overlap", pl.Float64),
    pl.Field("suffix_match", pl.Int8),

    pl.Field("miss_b_first", pl.Int8),
    pl.Field("miss_b_last", pl.Int8),
    pl.Field("miss_b_street1", pl.Int8),
    pl.Field("miss_b_city", pl.Int8),
    pl.Field("miss_b_state", pl.Int8),
    pl.Field("miss_b_zip5", pl.Int8),

    pl.Field("miss_x_first", pl.Int8),
    pl.Field("miss_x_last", pl.Int8),
    pl.Field("miss_x_street1", pl.Int8),
    pl.Field("miss_x_city", pl.Int8),
    pl.Field("miss_x_state", pl.Int8),
    pl.Field("miss_x_zip5", pl.Int8),

    pl.Field("sim_zip_num", pl.Float64),
    pl.Field("n_years_b", pl.Float64),
])


def _build_name_series(pb: pl.DataFrame, pa: pl.DataFrame, pc: pl.DataFrame) -> pl.Series:
    s_b = pb.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    s_a = pa.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_or_org_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    s_c = pc.select(
        pl.concat_str(
            [pl.col("first_name").cast(pl.Utf8).fill_null(""),
             pl.col("last_name").cast(pl.Utf8).fill_null("")],
            separator=" "
        ).alias("name")
    )["name"]

    return pl.concat([s_b, s_a, s_c], how="vertical")


# IDF built once from all provider names across A, B, C
_name_texts = _build_name_series(pb, pa, pc).drop_nulls().to_list()
idf_name = build_idf_from_texts(_name_texts)


def features_row(row: dict) -> dict:
    return _features_row(row, idf_name=idf_name)


# ============================================================
# BA features
# ============================================================
ba = (
    cand_ba
    .join(
        pb.select([
            "profile_id", "npi",
            pl.col("first_name").alias("b_first"),
            pl.col("last_name").alias("b_last"),
            pl.col("suffix").alias("b_suffix"),
            pl.col("street1").alias("b_street1"),
            pl.col("city").alias("b_city"),
            pl.col("state").alias("b_state"),
            pl.col("zip5").alias("b_zip5"),
            pl.col("n_years").cast(pl.Float64).fill_null(0.0).alias("b_n_years"),
        ]),
        on="profile_id",
        how="left",
    )
    .join(
        pa.select([
            pl.col("npi").alias("npi_a"),
            pl.col("first_name").alias("a_first"),
            pl.col("last_or_org_name").alias("a_last"),
            pl.col("credentials").alias("a_cred"),
            pl.col("street1").alias("a_street1"),
            pl.col("city").alias("a_city"),
            pl.col("state").alias("a_state"),
            pl.col("zip5").alias("a_zip5"),
        ]),
        on="npi_a",
        how="left",
    )
    .with_columns([
        pl.col("a_first").alias("x_first"),
        pl.col("a_last").alias("x_last"),
        pl.col("a_cred").alias("x_cred"),
        pl.lit("").alias("x_suffix"),
        pl.col("a_street1").alias("x_street1"),
        pl.col("a_city").alias("x_city"),
        pl.col("a_state").alias("x_state"),
        pl.col("a_zip5").alias("x_zip5"),
        pl.lit("").alias("b_cred"),
    ])
    .with_columns([
        pl.when(pl.col("npi").is_null())
          .then(pl.lit(None, dtype=pl.Int8))
          .when(pl.col("npi") == pl.col("npi_a"))
          .then(pl.lit(1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("label_weak_npi")
    ])
)

ba_feats = (
    ba.with_columns([
        pl.struct([
            "b_first","b_last","b_suffix","b_cred",
            "b_street1","b_city","b_state","b_zip5","b_n_years",
            "x_first","x_last","x_suffix","x_cred",
            "x_street1","x_city","x_state","x_zip5",
        ]).map_elements(features_row, return_dtype=FEATURE_STRUCT).alias("feat")
    ])
    .unnest("feat")
    .select([
        "profile_id","npi_a","pass","block_key","label_weak_npi",
        *[f.name for f in FEATURE_STRUCT.fields],
    ])
)

ba_feats.write_parquet(FEAT_DIR / "pair_features_ba.parquet")
print("Wrote:", FEAT_DIR / "pair_features_ba.parquet", ba_feats.shape)

# ============================================================
# BC features
# ============================================================
bc = (
    cand_bc
    .join(
        pb.select([
            "profile_id", "npi",
            pl.col("first_name").alias("b_first"),
            pl.col("last_name").alias("b_last"),
            pl.col("suffix").alias("b_suffix"),
            pl.col("street1").alias("b_street1"),
            pl.col("city").alias("b_city"),
            pl.col("state").alias("b_state"),
            pl.col("zip5").alias("b_zip5"),
            pl.col("n_years").cast(pl.Float64).fill_null(0.0).alias("b_n_years"),
        ]),
        on="profile_id",
        how="left",
    )
    .join(
        pc.select([
            pl.col("npi").alias("npi_c"),
            pl.col("first_name").alias("c_first"),
            pl.col("last_name").alias("c_last"),
            pl.col("state").alias("c_state"),
        ]),
        on="npi_c",
        how="left",
    )
    .with_columns([
        pl.col("c_first").alias("x_first"),
        pl.col("c_last").alias("x_last"),
        pl.lit("").alias("x_cred"),
        pl.lit("").alias("x_suffix"),
        pl.lit("").alias("x_street1"),
        pl.lit("").alias("x_city"),
        pl.col("c_state").alias("x_state"),
        pl.lit("").alias("x_zip5"),
        pl.lit("").alias("b_cred"),
    ])
    .with_columns([
        pl.when(pl.col("npi").is_null())
          .then(pl.lit(None, dtype=pl.Int8))
          .when(pl.col("npi") == pl.col("npi_c"))
          .then(pl.lit(1, dtype=pl.Int8))
          .otherwise(pl.lit(0, dtype=pl.Int8))
          .alias("label_weak_npi")
    ])
)

bc_feats = (
    bc.with_columns([
        pl.struct([
            "b_first","b_last","b_suffix","b_cred",
            "b_street1","b_city","b_state","b_zip5","b_n_years",
            "x_first","x_last","x_suffix","x_cred",
            "x_street1","x_city","x_state","x_zip5",
        ]).map_elements(features_row, return_dtype=FEATURE_STRUCT).alias("feat")
    ])
    .unnest("feat")
    .select([
        "profile_id","npi_c","pass","block_key","label_weak_npi",
        *[f.name for f in FEATURE_STRUCT.fields],
    ])
)

bc_feats.write_parquet(FEAT_DIR / "pair_features_bc.parquet")
print("Wrote:", FEAT_DIR / "pair_features_bc.parquet", bc_feats.shape)

print("Done. Feature files in:", FEAT_DIR)
