import html
import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger

# 上游目标地址
UPSTREAM = "https://api.deepseek.com"

# 内存中保留的最大请求条数 & 单条 body 最大存储长度（防止内存爆）
MAX_EXCHANGES = 200
MAX_BODY_CHARS = 200_000

# 响应侧不能照搬的 hop-by-hop header（由转发层自己重算）
HOP_BY_HOP = {
    "host",
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "proxy-authorization",
    "proxy-authenticate",
}

# 请求侧额外剔除：剥掉 Accept-Encoding 强制上游返回明文，
# 这样 aiter_raw() 拿到的就是未压缩字节，存进内存的 body 可直接阅读/渲染。
REQ_DROP = HOP_BY_HOP | {"accept-encoding"}


# ─────────────────────────── 内存存储 ───────────────────────────


@dataclass
class Exchange:
    id: str
    ts: datetime
    method: str
    url: str
    req_headers: dict
    req_body: str
    status: int | None = None
    resp_headers: dict = field(default_factory=dict)
    resp_body: str = ""
    is_stream: bool = False
    duration_ms: float | None = None
    truncated: bool = False

    def append_resp(self, text: str) -> None:
        if self.truncated:
            return
        room = MAX_BODY_CHARS - len(self.resp_body)
        if room <= 0:
            self.truncated = True
            return
        if len(text) > room:
            self.resp_body += text[:room]
            self.truncated = True
        else:
            self.resp_body += text


EXCHANGES: deque[Exchange] = deque(maxlen=MAX_EXCHANGES)


def fmt_body(text: str) -> str:
    """能解析成 JSON 就美化，否则原样返回。"""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text


# ─────────────────────────── 代理逻辑 ───────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM,
        timeout=httpx.Timeout(None, connect=10.0),
    )
    logger.info(f"代理已启动 -> {UPSTREAM}  |  查看面板: http://localhost:8000/__proxy")
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


async def do_proxy(path: str, request: Request) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    rid = uuid.uuid4().hex[:8]
    _log = logger.bind(req=rid)

    # 1. 解析进入的请求
    url = httpx.URL(path="/" + path, query=request.url.query.encode("utf-8"))
    headers = {k: v for k, v in request.headers.items() if k.lower() not in REQ_DROP}
    body = await request.body()
    req_body_text = body.decode("utf-8", errors="replace") if body else ""

    ex = Exchange(
        id=rid,
        ts=datetime.now(),
        method=request.method,
        url=str(url),
        req_headers=headers,
        req_body=req_body_text,
    )
    EXCHANGES.append(ex)

    _log.info(f"➡️  {request.method} {url}")
    _log.info(f"➡️  headers: {headers}")
    if req_body_text:
        _log.info(f"➡️  body:\n{fmt_body(req_body_text)}")

    # 2. 以流式方式发往上游
    started = time.perf_counter()
    upstream_req = client.build_request(
        method=request.method, url=url, headers=headers, content=body
    )
    upstream_resp = await client.send(upstream_req, stream=True)

    resp_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP_BY_HOP
    }
    content_type = upstream_resp.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type or "x-ndjson" in content_type

    ex.status = upstream_resp.status_code
    ex.resp_headers = resp_headers
    ex.is_stream = is_stream

    _log.info(f"⬅️  {upstream_resp.status_code}  ({content_type})")
    _log.info(f"⬅️  headers: {resp_headers}")

    if is_stream:
        # ─── 流式：逐块转发 + 累积存储 ───
        async def stream_body():
            n = 0
            try:
                async for chunk in upstream_resp.aiter_raw():
                    if chunk:
                        n += 1
                        text = chunk.decode("utf-8", errors="replace")
                        ex.append_resp(text)
                        if n <= 5 or n % 10 == 0:
                            _log.info(f"⬅️  chunk#{n}: {text[:300]!r}")
                    yield chunk
            finally:
                ex.duration_ms = (time.perf_counter() - started) * 1000
                _log.info(f"⬅️  流结束，共 {n} 块，{ex.duration_ms:.0f}ms")
                await upstream_resp.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type or None,
        )

    # ─── 非流式：一次读完 ───
    try:
        full = await upstream_resp.aread()
    finally:
        await upstream_resp.aclose()
    ex.duration_ms = (time.perf_counter() - started) * 1000
    text = full.decode("utf-8", errors="replace")
    ex.append_resp(text)
    _log.info(f"⬅️  body ({len(full)}B, {ex.duration_ms:.0f}ms):\n{fmt_body(text)}")

    return Response(
        content=full, status_code=upstream_resp.status_code, headers=resp_headers
    )


# ─────────────────────────── 查看面板 (HTML) ───────────────────────────


def _status_color(status: int | None) -> str:
    """报纸版克制用色：正常墨色，错误用红，跳转用灰。"""
    if status is None:
        return "var(--muted)"
    if status < 400:
        return "var(--ink)"
    return "var(--accent)"


PAGE_CSS = """
<style>
  :root{--paper:#f3efe4;--ink:#151515;--muted:#595247;--body:#2e2a24;--accent:#8a2f1b}
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--paper);color:var(--body)}
  body{font-family:"Songti SC","Noto Serif CJK SC","STSong",Georgia,serif;
       font-size:16px;line-height:1.8}
  a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--ink)}
  a:hover{color:var(--accent);border-color:var(--accent)}
  .page{max-width:1080px;margin:0 auto;padding:0 28px 60px}

  /* 报头 */
  .masthead{border-bottom:4px solid var(--ink);padding:26px 0 14px;margin-bottom:0}
  .masthead .meta{display:flex;justify-content:space-between;font-size:13px;
       color:var(--muted);letter-spacing:.08em;text-transform:uppercase}
  .masthead h1{font-weight:900;font-size:42px;letter-spacing:0;margin:10px 0 6px;color:var(--ink)}
  .masthead .subtitle{font-size:14px;color:var(--muted);margin:0}

  /* 栏目条 */
  .kicker{border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);
       font-weight:900;font-size:13px;letter-spacing:.18em;text-transform:uppercase;
       padding:6px 0;margin:22px 0 0;color:var(--ink)}

  /* 列表表格：结构全靠粗线 */
  table{width:100%;border-collapse:collapse;border-bottom:4px solid var(--ink)}
  th{text-align:left;font-weight:900;font-size:12px;letter-spacing:.12em;
     text-transform:uppercase;color:var(--ink);padding:10px 12px;
     border-bottom:2px solid var(--ink)}
  td{padding:11px 12px;border-bottom:1px solid rgba(21,21,21,.25);font-size:14px;vertical-align:top}
  tr:hover td{background:rgba(21,21,21,.05)}
  .badge{font-weight:900;color:var(--ink)}
  .stream{color:var(--accent);font-weight:800;font-size:12px;letter-spacing:.06em}

  /* 详情区块标题 + 正文框 */
  h2{font-weight:900;font-size:17px;color:var(--ink);margin:30px 0 8px;
     border-bottom:2px solid var(--ink);padding-bottom:4px;letter-spacing:.02em}
  pre{background:transparent;border:2px solid var(--ink);padding:14px 16px;
      overflow:auto;white-space:pre-wrap;word-break:break-word;
      font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;
      line-height:1.6;color:var(--ink)}
  .kv{color:var(--accent);font-weight:800}
  .no-data{color:var(--muted);padding:18px 0}

  @media(max-width:860px){.page{padding:0 16px 40px}.masthead h1{font-size:30px}}
</style>
"""


@app.get("/__proxy", response_class=HTMLResponse)
async def panel_list():
    rows = []
    for ex in reversed(EXCHANGES):
        dur = f"{ex.duration_ms:.0f}ms" if ex.duration_ms is not None else "—"
        stream = '<span class="stream">⚡stream</span>' if ex.is_stream else ""
        rows.append(
            f"<tr>"
            f"<td>{ex.ts:%H:%M:%S}</td>"
            f'<td class="badge">{ex.method}</td>'
            f'<td class="badge" style="color:{_status_color(ex.status)}">{ex.status or "—"}</td>'
            f"<td><a href='/__proxy/{ex.id}'>{html.escape(ex.url)}</a> {stream}</td>"
            f"<td>{dur}</td>"
            f"</tr>"
        )
    body = "".join(rows) or '<tr><td colspan=5 class="no-data">暂无请求</td></tr>'
    return f"""<!doctype html><html><head><meta charset=utf-8>{PAGE_CSS}</head><body>
    <div class="page">
      <header class="masthead">
        <div class="meta"><span>DEEPSEEK · 代理观察报</span><span>{datetime.now():%Y-%m-%d %H:%M}</span></div>
        <h1>请求观察日报</h1>
        <p class="subtitle">现存 {len(EXCHANGES)} 条 · 滚动保留最近 {MAX_EXCHANGES} 条 · 转发至 {UPSTREAM}</p>
      </header>
      <div class="kicker">实时往来记录</div>
      <table>
        <tr><th>时间</th><th>方法</th><th>状态</th><th>URL</th><th>耗时</th></tr>
        {body}
      </table>
    </div></body></html>"""


@app.get("/__proxy/{rid}", response_class=HTMLResponse)
async def panel_detail(rid: str):
    ex = next((e for e in EXCHANGES if e.id == rid), None)
    if ex is None:
        return HTMLResponse(
            f"<!doctype html><html><head><meta charset=utf-8>{PAGE_CSS}</head><body>"
            f'<div class="page"><p class="no-data">未找到该请求（可能已被滚动清除）。'
            f'<a href="/__proxy">返回列表</a></p></div></body></html>',
            status_code=404,
        )

    def render_headers(h: dict) -> str:
        return "".join(
            f'<span class="kv">{html.escape(k)}</span>: {html.escape(v)}\n'
            for k, v in h.items()
        )

    dur = f"{ex.duration_ms:.0f}ms" if ex.duration_ms is not None else "—"
    trunc = "  ⚠️ 已截断" if ex.truncated else ""
    return f"""<!doctype html><html><head><meta charset=utf-8>{PAGE_CSS}</head><body>
    <div class="page">
      <header class="masthead">
        <div class="meta"><span><a href="/__proxy">← 返回列表</a></span>
          <span>{ex.ts:%Y-%m-%d %H:%M:%S} · {dur} {'· ⚡STREAM' if ex.is_stream else ''}</span></div>
        <h1>{ex.method} <span style="color:{_status_color(ex.status)}">{ex.status}</span></h1>
        <p class="subtitle">{html.escape(ex.url)}</p>
      </header>

      <h2>请求 Headers</h2><pre>{render_headers(ex.req_headers)}</pre>
      <h2>请求 Body</h2><pre>{html.escape(fmt_body(ex.req_body)) or "(空)"}</pre>
      <h2>响应 Headers</h2><pre>{render_headers(ex.resp_headers)}</pre>
      <h2>响应 Body{trunc}</h2><pre>{html.escape(fmt_body(ex.resp_body)) or "(空)"}</pre>
    </div></body></html>"""


# ─── catch-all 代理路由（必须放在最后，否则会吞掉上面的面板路由）───


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(path: str, request: Request):
    return await do_proxy(path, request)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
