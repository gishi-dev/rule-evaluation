"""評価の計算。基準値・評価ランク・例外・抽出はすべて判定表で決め、ここでは数値の集計と受け渡しだけを行う。

達成率 = 3指標（稼働率・期日内完了率・満足度）それぞれの「実績 ÷ 基準」の平均。
1指標だけが突出して他の不足を打ち消さないよう、各指標の比は 1.2 を上限にする。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import duckdb
import pandas as pd

from .data import Inputs
from .rules import Match, Rulebook

RATIO_CAP = 1.2
RECENT_MONTHS = 3


@dataclass
class Evaluation:
    """1回の評価実行の結果。どのデータとどのルールの版で計算したかを持つ。"""

    executed_at: str
    period: list[str]
    rule_version: str
    data_fingerprint: str
    monthly: pd.DataFrame
    sites: pd.DataFrame
    people: pd.DataFrame
    percentiles: pd.DataFrame
    rankings: dict[str, pd.DataFrame]
    evidence: dict[str, list[dict]] = field(default_factory=dict)


def _months_between(start: str, end: str) -> int:
    """開設月・着任月を1か月目として、評価月までの月数を数える。"""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    return (ey - sy) * 12 + (em - sm) + 1


def _leave_ratio(start: str, end: str, period: list[str]) -> float:
    if not start or not end:
        return 0.0
    first = date.fromisoformat(f"{period[0]}-01")
    last_year, last_month = map(int, period[-1].split("-"))
    last = date(last_year + last_month // 12, last_month % 12 + 1, 1)  # 評価期間の翌月1日
    overlap = (min(date.fromisoformat(end), last) - max(date.fromisoformat(start), first)).days + 1
    return round(max(overlap, 0) / (last - first).days, 3)


def _metrics(inputs: Inputs) -> pd.DataFrame:
    return duckdb.query_df(
        inputs.monthly.merge(inputs.sites[["site_id", "処理枠_件月"]], on="site_id"),
        "m",
        """
        select site_id, 年月,
               完了件数::double / 処理枠_件月 as 稼働率,
               case when 完了件数 = 0 then 0 else 期日内完了件数::double / 完了件数 end as 期日内完了率,
               満足度平均 as 満足度
        from m
        """,
    ).df()


def evaluate(inputs: Inputs, rules: Rulebook) -> Evaluation:
    tables = rules.tables
    metrics = _metrics(inputs)
    period = sorted(metrics["年月"].unique())
    latest = period[-1]
    evidence: dict[str, list[dict]] = {}
    monthly_rows = []

    def note(key: str, month: str, match: Match) -> None:
        evidence.setdefault(key, []).append({"対象月": month, **asdict(match)})

    for _, site in inputs.sites.iterrows():
        for _, row in metrics[metrics["site_id"] == site["site_id"]].sort_values("年月").iterrows():
            months_open = _months_between(site["開設年月"], row["年月"])
            standard, match = tables["standards"].evaluate({"site": {"type": site["拠点種別"], "months_open": months_open}})
            note(site["site_id"], row["年月"], match)
            std = standard["standard"]
            ratios = [
                min(row["稼働率"] / std["utilization"], RATIO_CAP),
                min(row["期日内完了率"] / std["on_time"], RATIO_CAP),
                min(row["満足度"] / std["satisfaction"], RATIO_CAP),
            ]
            achievement = round(sum(ratios) / len(ratios), 4)
            ranked, match = tables["rank"].evaluate({"score": {"achievement": achievement}})
            note(site["site_id"], row["年月"], match)
            monthly_rows.append(
                {
                    "site_id": site["site_id"],
                    "拠点名": site["拠点名"],
                    "年月": row["年月"],
                    "稼働率": round(row["稼働率"], 3),
                    "期日内完了率": round(row["期日内完了率"], 3),
                    "満足度": row["満足度"],
                    "基準稼働率": std["utilization"],
                    "基準期日内完了率": std["on_time"],
                    "基準満足度": std["satisfaction"],
                    "総合達成率": achievement,
                    "評価": ranked["result"]["rank"],
                    "評価点": ranked["result"]["points"],
                }
            )
    monthly = pd.DataFrame(monthly_rows)

    people = inputs.people.set_index("担当")
    site_rows = []
    for _, site in inputs.sites.iterrows():
        own = monthly[monthly["site_id"] == site["site_id"]].sort_values("年月")
        manager = people.loc[site["site_id"]] if site["site_id"] in people.index else None
        average = round(own["総合達成率"].mean(), 4) if len(own) else 0.0
        ranked, match = tables["rank"].evaluate({"score": {"achievement": average}})
        note(site["site_id"], "6か月平均", match)
        context = {
            "site": {"months_open": _months_between(site["開設年月"], latest)},
            "data": {"missing_months": len(period) - len(own)},
            "manager": {
                "leave_ratio": _leave_ratio(manager["休職開始"], manager["休職終了"], period) if manager is not None else 0,
                "tenure_months": _months_between(manager["着任年月"], latest) if manager is not None else 0,
            },
        }
        exception, match = tables["exceptions"].evaluate(context)
        note(site["site_id"], "6か月平均", match)
        recent = own.tail(RECENT_MONTHS)
        summary = {
            "avg_points": round(own["評価点"].mean(), 2) if len(own) else 0,
            "recent_max_points": int(recent["評価点"].max()) if len(recent) == RECENT_MONTHS else 5,
        }
        target, match = tables["targets"].evaluate({"result": exception["result"], "summary": summary})
        note(site["site_id"], "6か月平均", match)
        site_rows.append(
            {
                "site_id": site["site_id"],
                "拠点名": site["拠点名"],
                "拠点種別": site["拠点種別"],
                "エリア": site["エリア"],
                "拠点長": manager["氏名"] if manager is not None else "",
                "評価した月数": len(own),
                "6か月平均の達成率": average,
                "6か月平均の評価": ranked["result"]["rank"],
                "6か月平均の評価点": summary["avg_points"],
                "判定区分": exception["result"]["status"],
                "判定理由": exception["result"]["reason"],
                "抽出区分": target["target"]["kind"],
                "抽出理由": target["target"]["reason"],
            }
        )
    sites = pd.DataFrame(site_rows)

    person_rows = []
    for _, person in inputs.people.iterrows():
        key = person["person_id"]
        if person["役割"] == "拠点長":
            row = sites[sites["site_id"] == person["担当"]].iloc[0]
            person_rows.append(
                {"person_id": key, "氏名": person["氏名"], "役割": "拠点長", "担当": row["拠点名"],
                 "評価の根拠": "担当拠点の6か月平均", "達成率": row["6か月平均の達成率"], "評価": row["6か月平均の評価"],
                 "判定区分": row["判定区分"], "判定理由": row["判定理由"], "抽出区分": row["抽出区分"]}
            )
            continue
        area = sites[(sites["エリア"] == person["担当"]) & (sites["判定区分"] == "通常")]
        average = round(area["6か月平均の達成率"].mean(), 4) if len(area) else 0.0
        ranked, match = tables["rank"].evaluate({"score": {"achievement": average}})
        note(key, "6か月平均", match)
        exception, match = tables["exceptions"].evaluate(
            {
                "site": {"months_open": 999},  # エリアマネージャーには開設月数の条件を当てない
                "data": {"missing_months": 0},
                "manager": {
                    "leave_ratio": _leave_ratio(person["休職開始"], person["休職終了"], period),
                    "tenure_months": _months_between(person["着任年月"], latest),
                },
            }
        )
        note(key, "6か月平均", match)
        person_rows.append(
            {"person_id": key, "氏名": person["氏名"], "役割": "エリアマネージャー", "担当": person["担当"],
             "評価の根拠": f"担当エリアの通常判定の拠点（{len(area)}拠点）の平均", "達成率": average,
             "評価": ranked["result"]["rank"], "判定区分": exception["result"]["status"],
             "判定理由": exception["result"]["reason"], "抽出区分": "対象外"}
        )
    people_result = pd.DataFrame(person_rows)

    return Evaluation(
        executed_at=datetime.now().isoformat(timespec="seconds"),
        period=period,
        rule_version=rules.version,
        data_fingerprint=inputs.fingerprint,
        monthly=monthly,
        sites=sites,
        people=people_result,
        percentiles=_percentiles(monthly, sites),
        rankings=_rankings(monthly, sites),
        evidence=evidence,
    )


def _percentiles(monthly: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """通常判定の拠点だけで、月ごとと6か月平均の達成率の分布を出す。"""
    normal = set(sites.loc[sites["判定区分"] == "通常", "site_id"])
    frame = pd.concat(
        [
            monthly[monthly["site_id"].isin(normal)][["年月", "総合達成率"]],
            sites[sites["site_id"].isin(normal)].assign(年月="6か月平均")[["年月", "6か月平均の達成率"]]
            .rename(columns={"6か月平均の達成率": "総合達成率"}),
        ]
    )
    return duckdb.query_df(
        frame,
        "f",
        """
        select 年月 as 対象, count(*) as 拠点数,
               round(quantile_cont(総合達成率, 0.10), 3) as p10,
               round(quantile_cont(総合達成率, 0.25), 3) as p25,
               round(quantile_cont(総合達成率, 0.50), 3) as 中央値,
               round(quantile_cont(総合達成率, 0.75), 3) as p75,
               round(quantile_cont(総合達成率, 0.90), 3) as p90
        from f group by 年月 order by 年月 = '6か月平均', 年月
        """,
    ).df()


def _rankings(monthly: pd.DataFrame, sites: pd.DataFrame, size: int = 3) -> dict[str, pd.DataFrame]:
    """月ごとと6か月平均の上位・下位。参考値・要確認の拠点は順位に入れない。"""
    normal = set(sites.loc[sites["判定区分"] == "通常", "site_id"])
    result = {}
    for month, group in monthly[monthly["site_id"].isin(normal)].groupby("年月"):
        ordered = group.sort_values("総合達成率", ascending=False)
        result[month] = pd.concat(
            [ordered.head(size).assign(区分="上位"), ordered.tail(size).iloc[::-1].assign(区分="下位")]
        )[["区分", "拠点名", "総合達成率", "評価"]]
    ordered = sites[sites["site_id"].isin(normal)].sort_values("6か月平均の達成率", ascending=False)
    result["6か月平均"] = pd.concat(
        [ordered.head(size).assign(区分="上位"), ordered.tail(size).iloc[::-1].assign(区分="下位")]
    )[["区分", "拠点名", "6か月平均の達成率", "6か月平均の評価"]].rename(
        columns={"6か月平均の達成率": "総合達成率", "6か月平均の評価": "評価"}
    )
    return result
