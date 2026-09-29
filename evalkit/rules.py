"""判定ルール（ZEN Engine の判定表）の読み書きと評価。

ルールは GoRules の JDM 形式（JSON）で `rules/` に置く。画面では表として編集し、
保存のたびに変更前の版を `data/rule_history/` へ残す。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import zen

DESCRIPTION = "説明"

# 表示順と、画面での呼び名。
RULE_FILES = {
    "standards": "基準値",
    "rank": "評価ランク",
    "exceptions": "例外条件",
    "targets": "対象者の抽出",
}


class RuleError(ValueError):
    """ルールの書き方の誤り、またはどの行にも当てはまらないこと。"""

    def __init__(self, messages: list[str]) -> None:
        super().__init__("\n".join(messages))
        self.messages = messages


def _syntax_error(expression: str, *, unary: bool) -> str | None:
    """書き方の誤りだけを返す。条件の欄は空欄（条件なし）を認め、結果の欄は空欄を認めない。

    ZEN の validate_unary_expression は判定表で使う「>= 12」も誤りと判定するため、
    仮の値で実際に評価して、構文と字句の誤りだけを拾う。
    """
    if not expression:
        return None if unary else "結果が空欄"
    try:
        if unary:
            zen.evaluate_unary_expression(expression, {"$": 0})
        else:
            zen.evaluate_expression(expression, {})
    except RuntimeError as error:
        try:
            kind = json.loads(str(error)).get("type")
        except ValueError:
            kind = None
        if kind in ("parserError", "lexerError"):
            return "式の書き方が正しくありません"
    return None


@dataclass
class Match:
    """どの表の何行目に、どの入力で当たったか。評価の根拠として残す。"""

    table: str
    row: int
    description: str
    inputs: dict
    outputs: dict


class RuleTable:
    def __init__(self, key: str, graph: dict) -> None:
        self.key = key
        self.graph = graph
        self.node = next(n for n in graph["nodes"] if n["type"] == "decisionTableNode")
        self._decision = zen.ZenEngine().create_decision(json.dumps(graph, ensure_ascii=False))

    @property
    def name(self) -> str:
        return RULE_FILES[self.key]

    @property
    def content(self) -> dict:
        return self.node["content"]

    def to_frame(self) -> pd.DataFrame:
        columns = self.content["inputs"] + self.content["outputs"]
        rows = [
            {DESCRIPTION: rule.get("_description", ""), **{c["name"]: rule.get(c["id"], "") for c in columns}}
            for rule in self.content["rules"]
        ]
        return pd.DataFrame(rows, columns=[DESCRIPTION, *[c["name"] for c in columns]]).fillna("")

    def with_frame(self, frame: pd.DataFrame) -> RuleTable:
        """画面で編集した表から、新しい判定表を作る。式の誤りがあれば RuleError にして保存させない。"""
        inputs, outputs = self.content["inputs"], self.content["outputs"]
        rules, errors = [], []
        for number, (_, row) in enumerate(frame.fillna("").iterrows(), 1):
            values = {c["id"]: str(row[c["name"]]).strip() for c in inputs + outputs}
            if not any(values.values()):
                continue
            for column in inputs:
                error = _syntax_error(values[column["id"]], unary=True)
                if error:
                    errors.append(f"{number}行目「{column['name']}」：{values[column['id']]}（{error}）")
            for column in outputs:
                error = _syntax_error(values[column["id"]], unary=False)
                if error:
                    errors.append(f"{number}行目「{column['name']}」：{values[column['id']] or '（空欄）'}（{error}）")
            rules.append({"_id": str(uuid.uuid4()), "_description": str(row[DESCRIPTION]).strip(), **values})
        if not rules:
            errors.append("ルールが1行もありません")
        if errors:
            raise RuleError(errors)
        graph = json.loads(json.dumps(self.graph))
        next(n for n in graph["nodes"] if n["type"] == "decisionTableNode")["content"]["rules"] = rules
        table = RuleTable(self.key, graph)
        table._decision.validate()
        return table

    def evaluate(self, context: dict) -> tuple[dict, Match]:
        response = self._decision.evaluate(context, {"trace": True})
        trace = response["trace"][self.node["id"]]["traceData"] or {}
        if "index" not in trace:
            # 当てはまる行がないまま空の結果で進めると、評価が欠けたまま出力される。
            raise RuleError([f"「{self.name}」に当てはまる行がありません（入力：{json.dumps(context, ensure_ascii=False, default=_plain)}）"])
        rule = trace.get("rule") or {}
        inputs = {c["name"]: _plain(_pick(context, c["field"])) for c in self.content["inputs"]}
        match = Match(
            table=self.name,
            row=trace["index"] + 1,
            description=rule.get("_description", "（該当する行なし）"),
            inputs=inputs,
            outputs=response["result"],
        )
        return response["result"], match


def _plain(value):
    """numpy の数値を、表示・保存できる通常の数値にする。"""
    return value.item() if hasattr(value, "item") else value


def _pick(context: dict, field: str):
    value = context
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


class Rulebook:
    def __init__(self, rules_dir: Path, history_dir: Path) -> None:
        self.rules_dir = rules_dir
        self.history_dir = history_dir
        self.tables = {
            key: RuleTable(key, json.loads((rules_dir / f"{key}.json").read_text(encoding="utf-8")))
            for key in RULE_FILES
        }

    @property
    def version(self) -> str:
        """ルール全体の版。評価結果に記録し、どのルールで計算したかを追えるようにする。"""
        digest = hashlib.sha256()
        for key in RULE_FILES:
            digest.update((self.rules_dir / f"{key}.json").read_bytes())
        return digest.hexdigest()[:8]

    def save(self, key: str, table: RuleTable, note: str) -> str:
        """変更前の版を履歴に残してから上書きする。新しい版を返す。"""
        before = self.version
        path = self.rules_dir / f"{key}.json"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        (self.history_dir / f"{stamp}_{key}_{before}.json").write_bytes(path.read_bytes())
        path.write_text(json.dumps(table.graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.tables[key] = table
        after = self.version
        with (self.history_dir / "changes.jsonl").open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {"saved_at": stamp, "table": RULE_FILES[key], "before": before, "after": after, "note": note},
                    ensure_ascii=False,
                )
                + "\n"
            )
        return after

    def changes(self) -> list[dict]:
        path = self.history_dir / "changes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
