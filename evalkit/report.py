"""評価結果のExcel書き出し。1回の評価を、根拠と実行条件ごと1ファイルにまとめる。"""

from __future__ import annotations

import io

import pandas as pd

from .engine import Evaluation


def evidence_frame(evaluation: Evaluation) -> pd.DataFrame:
    rows = []
    for key, matches in evaluation.evidence.items():
        for match in matches:
            rows.append(
                {
                    "対象": key,
                    "対象月": match["対象月"],
                    "判定表": match["table"],
                    "行": match["row"],
                    "当たったルール": match["description"],
                    "入力": ", ".join(f"{k}={v}" for k, v in match["inputs"].items()),
                }
            )
    return pd.DataFrame(rows)


def to_excel(evaluation: Evaluation) -> bytes:
    sites = evaluation.sites
    overview = pd.DataFrame(
        [
            ("実行日時", evaluation.executed_at),
            ("評価期間", f"{evaluation.period[0]}〜{evaluation.period[-1]}"),
            ("ルールの版", evaluation.rule_version),
            ("入力データの指紋", evaluation.data_fingerprint),
        ],
        columns=["項目", "値"],
    )
    rankings = pd.concat([frame.assign(対象=key) for key, frame in evaluation.rankings.items()])
    sheets = {
        "概要": overview,
        "拠点別評価": sites,
        "対象者別評価": evaluation.people,
        "業務改善報告書対象": sites[sites["抽出区分"] == "業務改善報告書"],
        "再教育対象": sites[sites["抽出区分"] == "再教育"],
        "要確認・参考値": sites[sites["判定区分"] != "通常"],
        "月別評価": evaluation.monthly,
        "パーセンタイル": evaluation.percentiles,
        "上位・下位": rankings[["対象", "区分", "拠点名", "総合達成率", "評価"]],
        "算出根拠": evidence_frame(evaluation),
    }
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for column in sheet.columns:
                width = max(len(str(cell.value or "")) for cell in column)
                sheet.column_dimensions[column[0].column_letter].width = min(max(width * 1.8, 8), 60)
    return buffer.getvalue()
