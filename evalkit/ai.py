"""月次レポートの所見の下書きを、Claude APIで作る。

評価の計算にはAIを使わない。AIに渡すのは計算済みの集計だけで、所見は人が確認してから使う。
"""

from __future__ import annotations

import json

import anthropic

from .engine import Evaluation

INSTRUCTIONS = """\
あなたは、多拠点のサービス事業者で月次の拠点評価をまとめる担当です。
渡された評価結果の集計から、月次レポートに載せる所見の下書きを日本語で作ってください。

- 数値は、渡されたデータにあるものだけを引用する。新しい計算、推測、渡されていない原因の断定をしない。
- 賞与・昇降格・処分などの人事判断を書かない。評価結果をどう扱うかは人が決める。
- 個人名は書かず、拠点名で書く。
- 次の順に、それぞれ2〜3文で書く。見出しは「### 全体の傾向」「### 目立つ拠点」「### 確認が必要な事項」の3つだけにし、全体のタイトルは付けない。
- 「確認が必要な事項」には、参考値・要確認の拠点と、入力チェックで止まったデータの件数を書く。
"""


def _summary(evaluation: Evaluation, issue_count: int) -> dict:
    sites = evaluation.sites
    return {
        "評価期間": f"{evaluation.period[0]}〜{evaluation.period[-1]}",
        "拠点数": len(sites),
        "6か月平均の評価の分布": sites["6か月平均の評価"].value_counts().to_dict(),
        "達成率のパーセンタイル": evaluation.percentiles.to_dict(orient="records"),
        "上位・下位（直近月）": evaluation.rankings[evaluation.period[-1]].to_dict(orient="records"),
        "上位・下位（6か月平均）": evaluation.rankings["6か月平均"].to_dict(orient="records"),
        "抽出された拠点": sites[sites["抽出区分"] != "対象外"][["拠点名", "抽出区分", "抽出理由"]].to_dict(orient="records"),
        "参考値・要確認の拠点": sites[sites["判定区分"] != "通常"][["拠点名", "判定区分", "判定理由"]].to_dict(orient="records"),
        "入力チェックで止まったデータの件数": issue_count,
    }


def draft_commentary(client: anthropic.Anthropic, evaluation: Evaluation, issue_count: int, *, model: str) -> str:
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": json.dumps(_summary(evaluation, issue_count), ensure_ascii=False)}],
        output_config={"effort": "medium"},
        # 安全上の理由で断られた場合に、別モデルで答え直す（サーバー側フォールバック）。
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("所見の下書きを作れませんでした。")
    return "".join(block.text for block in response.content if block.type == "text").strip()
