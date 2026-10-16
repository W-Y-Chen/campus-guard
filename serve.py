"""零依赖 Web 服务（仅用 Python 标准库）。

    python serve.py
    # 打开 http://127.0.0.1:8765

与 core 引擎共用同一套逻辑；如需异步框架可自行包装，接口路径保持一致。
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agent.loop import AgentLoop                      # noqa: E402
from bench.dataset import build                       # noqa: E402
from core.evolver import Evolver                      # noqa: E402
from core.mutation import generate, report            # noqa: E402
from core.pipeline import Guard                       # noqa: E402
from core.reporter import Reporter                    # noqa: E402
from core.types import Action, Sens, Trust            # noqa: E402
from bench.evaluate import calibrate_operating_points  # noqa: E402

PORT = 8765

print("[boot] 正在构建数据集与训练语义通道 ...")
DATA = build()
GUARD = Guard()
GUARD.semantic.fit(DATA["train_texts"], DATA["train_labels"])
GUARD.semantic.calibrate(DATA["val_texts"], DATA["val_labels"], iters=60)
CALIB = calibrate_operating_points(GUARD, DATA)
LOOP = AgentLoop(GUARD)
print(f"[boot] 就绪。工作点标定：{CALIB}")


def _json(handler, obj: dict, code: int = 200) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):        # 静音默认访问日志
        pass

    # ---------------- GET ----------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/console"):
            self._serve_file(ROOT / "console" / "index.html", "text/html")
        elif path == "/api/health":
            _json(self, {"ok": True, "calibration": CALIB,
                         "dataset": DATA["meta"]})
        elif path == "/api/ledger":
            _json(self, {"verify": GUARD.ledger.verify(),
                         "count": len(GUARD.ledger.records),
                         "records": [r.to_dict()
                                     for r in GUARD.ledger.records[-200:]]})
        elif path == "/api/reports":
            _json(self, Reporter(GUARD.ledger).all())
        elif path == "/api/samples":
            _json(self, {
                "attacks": [p for _, p in DATA["test_seeds"]],
                "benign": DATA["benign_text"][:12],
            })
        else:
            self.send_error(404)

    # ---------------- POST ----------------

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/detect":
            self._detect(payload)
        elif path == "/api/session":
            self._session(payload)
        elif path == "/api/mutate":
            self._mutate(payload)
        elif path == "/api/evolve":
            self._evolve(payload)
        elif path == "/api/tamper":
            self._tamper(payload)
        else:
            self.send_error(404)

    # ---------------- 具体实现 ----------------

    def _detect(self, p: dict) -> None:
        text = p.get("text", "")
        trust = Trust(int(p.get("trust", 0)))
        sens = Sens(int(p.get("sens", 3)))
        _json(self, GUARD.inspect(text, trust, sens))

    def _session(self, p: dict) -> None:
        msg = p.get("user_msg", "帮我查一下选课时间")
        docs = [(d.get("id", f"web:{i}"), d.get("text", ""))
                for i, d in enumerate(p.get("docs", []))]
        res = LOOP.run(msg, external_docs=docs,
                       session_id=p.get("session_id", "web"))
        res["ledger_verify"] = GUARD.ledger.verify()
        _json(self, res)

    def _mutate(self, p: dict) -> None:
        combo = int(p.get("combo", 6))
        triple = int(p.get("triple", 6))
        mutants = generate(DATA["test_seeds"], k=2, combo=combo, triple=triple)
        for m in mutants:
            if not m.valid:
                continue
            d = GUARD.detect(m.payload, Trust.UNTRUSTED, Sens.L3, "mutant")
            m.killed = d.action in (Action.BLOCK, Action.QUARANTINE)
            m.killed_by = d.action.value
        r = report(mutants)
        _json(self, r.to_dict())

    def _evolve(self, p: dict) -> None:
        rounds = int(p.get("rounds", 3))
        ev = Evolver(GUARD, DATA)
        _json(self, {"history": ev.run(rounds=rounds, k=2)})

    def _tamper(self, p: dict) -> None:
        """演示：篡改/删除一条审计记录后，校验立刻失败。"""
        seq = int(p.get("seq", 0))
        mode = p.get("mode", "modify")
        if not GUARD.ledger.records:
            _json(self, {"ok": False, "msg": "暂无审计记录"})
            return
        seq = min(seq, len(GUARD.ledger.records) - 1)
        if mode == "delete":
            del GUARD.ledger.records[seq]
        else:
            GUARD.ledger.records[seq].decision = "pass"
        _json(self, {"tampered_seq": seq, "mode": mode,
                     "verify": GUARD.ledger.verify()})

    # ---------------- 静态文件 ----------------

    def _serve_file(self, path: Path, ctype: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n  校盾 CampusGuard 控制台： http://127.0.0.1:{PORT}\n"
          f"  按 Ctrl+C 停止\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
