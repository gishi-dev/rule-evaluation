"""入力データの読み込みと検査。検査で止まった行は評価に使わず、一覧にして人が確認する。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

MONTH = r"^\d{4}-(0[1-9]|1[0-2])$"
FILES = ("sites.csv", "people.csv", "monthly.csv")


@dataclass
class Inputs:
    sites: pd.DataFrame
    people: pd.DataFrame
    monthly: pd.DataFrame
    issues: pd.DataFrame
    fingerprint: str


def _sites_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "site_id": pa.Column(str, unique=True),
            "拠点名": pa.Column(str),
            "拠点種別": pa.Column(str, pa.Check.isin(["大型センター", "標準拠点", "サテライト"])),
            "エリア": pa.Column(str),
            "開設年月": pa.Column(str, pa.Check.str_matches(MONTH)),
            "処理枠_件月": pa.Column(int, pa.Check.gt(0)),
        }
    )


def _people_schema(site_ids: set[str], areas: set[str]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "person_id": pa.Column(str, unique=True),
            "氏名": pa.Column(str),
            "役割": pa.Column(str, pa.Check.isin(["拠点長", "エリアマネージャー"])),
            "担当": pa.Column(str, pa.Check.isin(site_ids | areas, error="担当の拠点・エリアが存在しない")),
            "着任年月": pa.Column(str, pa.Check.str_matches(MONTH)),
            "休職開始": pa.Column(str, nullable=True),
            "休職終了": pa.Column(str, nullable=True),
        }
    )


def _monthly_schema(site_ids: set[str]) -> pa.DataFrameSchema:
    count = [pa.Check.ge(0, error="件数がマイナス")]
    return pa.DataFrameSchema(
        {
            "site_id": pa.Column(str, pa.Check.isin(site_ids, error="拠点一覧にない拠点ID")),
            "年月": pa.Column(str, pa.Check.str_matches(MONTH)),
            "受付件数": pa.Column(int, count),
            "完了件数": pa.Column(int, count),
            "期日内完了件数": pa.Column(int, count),
            "満足度平均": pa.Column(float, pa.Check.in_range(1, 5, error="満足度が1〜5の範囲外")),
            "スタッフ数": pa.Column(int, pa.Check.gt(0)),
        },
        checks=[
            pa.Check(lambda df: df["完了件数"] <= df["受付件数"], error="完了件数が受付件数を超えている"),
            pa.Check(lambda df: df["期日内完了件数"] <= df["完了件数"], error="期日内完了件数が完了件数を超えている"),
        ],
        unique=["site_id", "年月"],
    )


def _validate(name: str, frame: pd.DataFrame, schema: pa.DataFrameSchema) -> tuple[pd.DataFrame, list[dict]]:
    try:
        return schema.validate(frame, lazy=True), []
    except pa.errors.SchemaErrors as errors:
        cases = errors.failure_cases
        issues: dict[tuple, dict] = {}
        for _, case in cases.iterrows():
            # 表全体の検査は列ごとに失敗が並ぶため、行と検査の組で1件にまとめる。
            whole_row = case["schema_context"] == "DataFrameSchema"
            row = int(case["index"]) + 2 if pd.notna(case["index"]) else None  # CSVの見出し行を1行目とする
            issues.setdefault(
                (row, str(case["check"])),
                {
                    "ファイル": name,
                    "行": row,
                    "列": "" if whole_row else str(case["column"]),
                    "内容": str(case["check"]),
                    "値": "" if whole_row else str(case["failure_case"]),
                },
            )
        issues = list(issues.values())
        bad_rows = {int(i) for i in cases["index"].dropna()}
        return frame.drop(index=[i for i in bad_rows if i in frame.index]), issues


def load_inputs(data_dir: Path) -> Inputs:
    digest = hashlib.sha256()
    for name in FILES:
        digest.update((data_dir / name).read_bytes())
    read = lambda name: pd.read_csv(data_dir / name, dtype_backend="numpy_nullable", keep_default_na=False)  # noqa: E731
    sites_raw = read("sites.csv").astype({"site_id": str, "開設年月": str})
    sites, issues = _validate("sites.csv", sites_raw, _sites_schema())
    people, people_issues = _validate(
        "people.csv", read("people.csv").astype(str), _people_schema(set(sites["site_id"]), set(sites["エリア"]))
    )
    monthly_raw = read("monthly.csv").astype({"site_id": str, "年月": str, "満足度平均": float})
    monthly, monthly_issues = _validate("monthly.csv", monthly_raw, _monthly_schema(set(sites["site_id"])))
    all_issues = pd.DataFrame(issues + people_issues + monthly_issues, columns=["ファイル", "行", "列", "内容", "値"])
    return Inputs(sites, people, monthly, all_issues, digest.hexdigest()[:8])
