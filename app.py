"""拠点評価の自動化（デモ）の画面。

起動: streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evalkit.ai import draft_commentary
from evalkit.data import load_inputs
from evalkit.engine import Evaluation, evaluate
from evalkit.report import evidence_frame, to_excel
from evalkit.rules import RULE_FILES, Rulebook, RuleError

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / os.getenv("RULE_EVAL_DATA_DIR", "sample")
RULES_DIR = ROOT / "rules"
STATE_DIR = ROOT / "data"
MODEL = os.getenv("RULE_EVAL_MODEL", "claude-opus-5")


def rulebook() -> Rulebook:
    return Rulebook(RULES_DIR, STATE_DIR / "rule_history")


def run_evaluation() -> None:
    inputs = load_inputs(DATA_DIR)
    try:
        result = evaluate(inputs, rulebook())
    except RuleError as error:
        # 評価が欠けたまま結果を差し替えない。直前の結果を残して、原因を表示する。
        st.session_state.run_error = error.messages
        return
    st.session_state.pop("run_error", None)
    st.session_state.evaluation = result
    st.session_state.issues = inputs.issues
    st.session_state.pop("commentary", None)
    STATE_DIR.mkdir(exist_ok=True)
    with (STATE_DIR / "runs.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "実行日時": result.executed_at,
                    "評価期間": f"{result.period[0]}〜{result.period[-1]}",
                    "ルールの版": result.rule_version,
                    "入力データの指紋": result.data_fingerprint,
                    "業務改善報告書": int((result.sites["抽出区分"] == "業務改善報告書").sum()),
                    "再教育": int((result.sites["抽出区分"] == "再教育").sum()),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def runs() -> list[dict]:
    path = STATE_DIR / "runs.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x] if path.exists() else []


def sites_tab(result: Evaluation) -> None:
    columns = ["拠点名", "拠点種別", "エリア", "6か月平均の達成率", "6か月平均の評価", "判定区分", "抽出区分"]
    # 対応が必要な拠点を先頭に並べる。
    order = {"業務改善報告書": 0, "再教育": 1}
    sites = result.sites.assign(
        _order=[order.get(t, 2 if s == "要確認" else 3 if s == "参考値" else 4)
                for t, s in zip(result.sites["抽出区分"], result.sites["判定区分"])]
    ).sort_values(["_order", "site_id"])
    event = st.dataframe(
        sites[columns],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={"6か月平均の達成率": st.column_config.NumberColumn(format="%.3f")},
        key="sites",
    )
    if not event.selection.rows:
        st.caption("行を選ぶと、月ごとの評価と算出根拠（どの判定表のどのルールに当たったか）が表示されます。")
        return
    site = sites.iloc[event.selection.rows[0]]
    st.subheader(site["拠点名"])
    st.caption(f"判定区分：{site['判定区分']}（{site['判定理由']}）　抽出区分：{site['抽出区分']} {site['抽出理由']}")
    monthly = result.monthly[result.monthly["site_id"] == site["site_id"]]
    st.markdown("**月ごとの評価**")
    st.dataframe(
        monthly[["年月", "稼働率", "基準稼働率", "期日内完了率", "基準期日内完了率", "満足度", "基準満足度", "総合達成率", "評価"]],
        hide_index=True,
        width="stretch",
    )
    chart, _ = st.columns(2)
    chart.markdown("**総合達成率の推移**")
    chart.line_chart(monthly.set_index("年月")["総合達成率"], height=220)
    st.markdown("**算出根拠**")
    evidence = evidence_frame(result)
    st.dataframe(evidence[evidence["対象"] == site["site_id"]].drop(columns="対象"), hide_index=True, width="stretch")


def targets_tab(result: Evaluation) -> None:
    sites = result.sites
    for title, frame, columns in (
        ("業務改善報告書の対象", sites[sites["抽出区分"] == "業務改善報告書"], ["拠点名", "拠点長", "6か月平均の評価点", "抽出理由"]),
        ("再教育の対象", sites[sites["抽出区分"] == "再教育"], ["拠点名", "拠点長", "6か月平均の評価点", "抽出理由"]),
        ("要確認・参考値（自動では判定しない拠点）", sites[sites["判定区分"] != "通常"], ["拠点名", "拠点長", "判定区分", "判定理由"]),
    ):
        st.markdown(f"**{title}**（{len(frame)}件）")
        if len(frame):
            st.dataframe(frame[columns], hide_index=True, width="stretch")
        else:
            st.caption("該当なし")
    st.caption("抽出結果は候補です。報告書の依頼や再教育の実施は、担当者が内容を確認してから決めます。")


def report_tab(result: Evaluation) -> None:
    st.markdown("**総合達成率のパーセンタイル**（通常判定の拠点のみ）")
    st.dataframe(result.percentiles, hide_index=True, width="stretch")
    target = st.selectbox("上位・下位3位の対象", list(result.rankings)[::-1])
    st.dataframe(result.rankings[target], hide_index=True, width="stretch")
    st.divider()
    st.markdown("**所見の下書き（AI）**")
    st.caption("AIは計算済みの集計だけを読み、所見の文章を下書きします。評価の計算にはAIを使っていません。")
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if st.button("所見の下書きを作る", disabled=not has_key):
        with st.spinner("下書きを作っています…"):
            try:
                st.session_state.commentary = draft_commentary(
                    anthropic.Anthropic(), result, len(st.session_state.issues), model=MODEL
                )
            except (anthropic.APIError, RuntimeError) as error:
                st.error(f"下書きを作れませんでした：{error}")
    if not has_key:
        st.caption(".env に ANTHROPIC_API_KEY を設定すると使えます。")
    if "commentary" in st.session_state:
        with st.container(border=True):
            # 下書きの見出しは画面の見出しより目立つため、記号の数によらず小さい見出しにそろえる。
            st.markdown(re.sub(r"(?m)^#{1,6}\s+", "##### ", st.session_state.commentary))
        with st.expander("下書きを編集する"):
            st.session_state.commentary = st.text_area("所見", st.session_state.commentary, height=320)


def issues_tab() -> None:
    issues = st.session_state.issues
    st.caption("検査で止まった行は評価に使いません。その拠点に欠けた月ができるため、例外条件で「要確認」になります。")
    if issues.empty:
        st.write("問題はありませんでした。")
    else:
        st.dataframe(issues, hide_index=True, width="stretch")


def rules_tab() -> None:
    book = rulebook()
    key = st.selectbox("判定表", list(RULE_FILES), format_func=RULE_FILES.get)
    table = book.tables[key]
    st.caption(
        "上の行から順に調べ、最初に当てはまった行を使います。空欄は「条件なし」です。"
        "数値は「>= 12」「< 0.85」「[0.92..0.98)」、文字は「\"大型センター\"」のように書きます。"
    )
    edited = st.data_editor(table.to_frame(), num_rows="dynamic", width="stretch", key=f"editor-{key}")
    note = st.text_input("変更の理由", placeholder="例：サテライトの基準稼働率を見直し")
    if st.button("ルールを保存", type="primary", disabled=not note.strip()):
        try:
            version = book.save(key, table.with_frame(edited), note.strip())
        except RuleError as error:
            st.error("保存できませんでした。次の箇所を直してください。\n\n" + "\n".join(f"- {m}" for m in error.messages))
        else:
            st.success(f"保存しました（ルールの版：{version}）。次に評価を実行したときから反映されます。")
    changes = book.changes()
    if changes:
        st.markdown("**変更履歴**")
        st.dataframe(pd.DataFrame(changes[::-1]).rename(columns={
            "saved_at": "保存日時", "table": "判定表", "before": "変更前の版", "after": "変更後の版", "note": "理由"}),
            hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="拠点評価の自動化（デモ）", page_icon="📊", layout="wide")
    if "evaluation" not in st.session_state:
        run_evaluation()
    if "evaluation" not in st.session_state:
        st.error("評価を実行できませんでした。\n\n" + "\n".join(f"- {m}" for m in st.session_state.run_error))
        return
    result: Evaluation = st.session_state.evaluation

    with st.sidebar:
        st.title("拠点評価の自動化")
        st.caption("架空の家電メーカー「ギシ電機」の修理・サポート拠点（40拠点）の実績で動くデモです。")
        if st.button("評価を実行", type="primary", width="stretch"):
            run_evaluation()
            st.rerun()
        st.download_button(
            "Excelに書き出す",
            to_excel(result),
            file_name=f"拠点評価_{result.period[-1]}.xlsx",
            width="stretch",
        )
        st.divider()
        st.caption(
            f"評価期間：{result.period[0]}〜{result.period[-1]}\n\n"
            f"ルールの版：{result.rule_version}　データの指紋：{result.data_fingerprint}\n\n"
            f"実行日時：{result.executed_at.replace('T', ' ')}"
        )
        history = runs()
        if history:
            with st.expander(f"実行履歴（{len(history)}回）"):
                st.dataframe(pd.DataFrame(history[::-1]).head(10), hide_index=True)

    st.header("ギシ電機 拠点評価（月次）")
    if "run_error" in st.session_state:
        st.error("評価を実行できませんでした。表示は前回の結果です。\n\n" + "\n".join(f"- {m}" for m in st.session_state.run_error))
    sites = result.sites
    a, b, c, d, e = st.columns(5)
    a.metric("評価した拠点", len(sites))
    b.metric("業務改善報告書", int((sites["抽出区分"] == "業務改善報告書").sum()))
    c.metric("再教育", int((sites["抽出区分"] == "再教育").sum()))
    d.metric("要確認・参考値", int((sites["判定区分"] != "通常").sum()))
    e.metric("入力チェックで止めた行", len(st.session_state.issues))

    tabs = st.tabs(["拠点別評価", "対象者別評価", "抽出一覧", "月次レポート", "入力チェック", "評価ルール"])
    with tabs[0]:
        sites_tab(result)
    with tabs[1]:
        st.dataframe(result.people, hide_index=True, width="stretch",
                     column_config={"達成率": st.column_config.NumberColumn(format="%.3f")})
    with tabs[2]:
        targets_tab(result)
    with tabs[3]:
        report_tab(result)
    with tabs[4]:
        issues_tab()
    with tabs[5]:
        rules_tab()


main()
