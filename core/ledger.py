"""M9 防篡改审计链：哈希链 + Merkle 锚定 + 校验。

设计：
  H_i = SHA256(H_{i-1} || canonical_json(record_i))
每 128 条计算一次 Merkle 根写入锚点表。表为只追加（append-only）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field


def canonical_json(obj) -> str:
    """确定性序列化：键排序 + 固定分隔符，否则哈希不可复现。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    seq: int
    ts: str
    session_id: str
    stage: str
    actor: str
    payload_ref: str          # 大文本只存 SHA256，不落明文，避免日志本身泄露
    decision: str
    evidence: dict = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""

    def body(self) -> dict:
        d = asdict(self)
        d.pop("hash", None)
        return d

    def compute_hash(self) -> str:
        return sha256(self.prev_hash + canonical_json(self.body()))

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


GENESIS = "0" * 64
ANCHOR_INTERVAL = 128


class Ledger:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []
        self.anchors: list[dict] = []
        self._by_session: dict[str, list[int]] = {}

    # ---- 写入 ----

    def append(self, session_id: str, stage: str, actor: str,
               payload: str | None, decision: str,
               evidence: dict | None = None, ts: str | None = None) -> AuditRecord:
        prev = self.records[-1].hash if self.records else GENESIS
        rec = AuditRecord(
            seq=len(self.records),
            ts=ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            session_id=session_id,
            stage=stage,
            actor=actor,
            payload_ref=sha256(payload) if payload is not None else "",
            decision=decision,
            evidence=evidence or {},
            prev_hash=prev,
        )
        rec.hash = rec.compute_hash()
        self.records.append(rec)
        self._by_session.setdefault(session_id, []).append(rec.seq)

        if len(self.records) % ANCHOR_INTERVAL == 0:
            self._seal_anchor()
        return rec

    def _seal_anchor(self) -> None:
        start = max(0, len(self.records) - ANCHOR_INTERVAL)
        chunk = self.records[start:len(self.records)]
        self.anchors.append({
            "from_seq": chunk[0].seq,
            "to_seq": chunk[-1].seq,
            "merkle_root": merkle_root([r.hash for r in chunk]),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # ---- 查询 ----

    def session(self, session_id: str) -> list[AuditRecord]:
        return [self.records[i] for i in self._by_session.get(session_id, [])]

    def export(self) -> dict:
        return {
            "count": len(self.records),
            "genesis": GENESIS,
            "records": [r.to_dict() for r in self.records],
            "anchors": self.anchors,
        }

    # ---- 校验 ----

    def verify(self) -> dict:
        """逐条重算哈希；返回首个不一致的位置。"""
        prev = GENESIS
        for i, r in enumerate(self.records):
            if r.prev_hash != prev:
                return {"ok": False, "first_bad_seq": i, "reason": "prev_hash 断裂",
                        "checked": i}
            if r.hash != r.compute_hash():
                return {"ok": False, "first_bad_seq": i, "reason": "记录被篡改",
                        "checked": i}
            prev = r.hash
        return {"ok": True, "first_bad_seq": None, "checked": len(self.records)}

    def verify_anchors(self) -> dict:
        bad = []
        for a in self.anchors:
            chunk = self.records[a["from_seq"]:a["to_seq"] + 1]
            if merkle_root([r.hash for r in chunk]) != a["merkle_root"]:
                bad.append(a["from_seq"])
        return {"ok": not bad, "bad_anchors": bad}


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return sha256("")
    level = list(hashes)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(sha256(left + right))
        level = nxt
    return level[0]
