#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pilot_luna_fresh2_personal"
DIST = ROOT / "vis/dist"
MEDIA_PATTERN = re.compile(r"(?:contact|evidence)-[^\"'\s<>]+?\.(?:jpg|jpeg|png|webp)", re.I)
SECRET_PATTERN = re.compile(r"(?i)(?:sk-|bearer\s+|api[_-]?key[\"'=:\s]+)[A-Za-z0-9_.-]{12,}")


def clean(value):
    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
            if key.lower() not in {"encrypted_content", "authorization", "api_key", "apikey"}
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<embedded-image-removed>", value)
        return SECRET_PATTERN.sub("<secret-redacted>", value)
    return value


def segment_number(path):
    match = re.search(r"segment-(\d+)", path.name)
    return int(match.group(1)) if match else 10_000


def encode_media(job):
    source, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", "scale='min(960,iw)':-2", "-c:v", "libwebp", "-quality", "55",
            str(target),
        ],
        check=True,
    )


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "media").mkdir(parents=True)

    rollouts = []
    media_jobs = []
    all_results = sorted(RUN.glob("train/epoch-*/batch-*/question-*/rollout-*/result.json"))
    for result_path in all_results:
        result = clean(json.loads(result_path.read_text()))
        parts = result_path.parts
        epoch = next(part for part in parts if part.startswith("epoch-"))
        batch = next(part for part in parts if part.startswith("batch-"))
        question_dir = next(part for part in parts if part.startswith("question-"))
        rollout_dir = next(part for part in parts if part.startswith("rollout-"))
        rollout_root = result_path.parent
        attempts = []
        rollout_media = []

        for attempt_dir in sorted(rollout_root.glob("attempt-*")):
            media_by_name = {path.name: path for path in attempt_dir.glob("media/*") if path.is_file()}
            traces = []
            linked_names = set()
            trace_paths = sorted(attempt_dir.glob("traces/*.json"), key=lambda path: (segment_number(path), path.name))
            for trace_path in trace_paths:
                trace = clean(json.loads(trace_path.read_text()))
                serialized = json.dumps(trace, ensure_ascii=False)
                refs = sorted(set(MEDIA_PATTERN.findall(serialized)))
                clip_match = re.search(r"clip-([0-9a-f]+)-segment", trace_path.name)
                if clip_match:
                    refs = sorted(set(refs) | {name for name in media_by_name if clip_match.group(1) in name})
                linked_names.update(name for name in refs if name in media_by_name)
                traces.append({
                    "file": trace_path.name,
                    "kind": trace_path.name.split("-segment-")[0],
                    "segment": segment_number(trace_path),
                    "input": trace.get("input"),
                    "is_complete": trace.get("is_complete"),
                    "final_output": trace.get("final_output"),
                    "history": trace.get("history", []),
                    "media_refs": refs,
                })

            error_path = attempt_dir / "error.json"
            error = clean(json.loads(error_path.read_text())) if error_path.exists() else None
            for name, source in sorted(media_by_name.items()):
                relative = Path(epoch) / batch / question_dir / rollout_dir / attempt_dir.name / f"{Path(name).stem}.webp"
                media_jobs.append((source, DIST / "media" / relative))
                match = re.match(r"(?:contact|evidence)-([0-9.]+)", name)
                rollout_media.append({
                    "source_name": name,
                    "url": f"media/{relative.as_posix()}",
                    "kind": "evidence" if name.startswith("evidence-") else "contact",
                    "timestamp_s": float(match.group(1)) if match else None,
                    "attempt": attempt_dir.name,
                    "linked": name in linked_names,
                })
            attempts.append({"name": attempt_dir.name, "error": error, "traces": traces})

        rollouts.append({
            "id": f"{epoch}/{batch}/{question_dir}/{rollout_dir}",
            "epoch": int(epoch.split("-")[1]),
            "batch": int(batch.split("-")[1]),
            "question_index": result["question_index"],
            "rollout": int(rollout_dir.split("-")[1]),
            "rollout_id": result["rollout_id"],
            "status": result["status"],
            "answer": result["answer"],
            "reference_answer": result["reference_answer"],
            "score": result["score"],
            "reflection": result["reflection"],
            "meta_reflection": result.get("meta_reflection"),
            "attempts": attempts,
            "media": rollout_media,
        })

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        list(pool.map(encode_media, media_jobs))

    payload = {
        "run": "pilot_luna_fresh2_personal",
        "model": "gpt-5.6-luna",
        "rollout_count": len(rollouts),
        "trace_count": sum(len(attempt["traces"]) for rollout in rollouts for attempt in rollout["attempts"]),
        "media_count": sum(len(rollout["media"]) for rollout in rollouts),
        "rollouts": rollouts,
    }
    (DIST / "data.js").write_text("window.TRACE_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (DIST / "index.html").write_text(INDEX_HTML)
    (DIST / "styles.css").write_text(STYLES)
    (DIST / "app.js").write_text(APP_JS)
    print(json.dumps({
        "rollouts": payload["rollout_count"],
        "traces": payload["trace_count"],
        "media": payload["media_count"],
        "dist_bytes": sum(path.stat().st_size for path in DIST.rglob("*") if path.is_file()),
    }))


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Luna Pilot · Rollout Trace Explorer</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="shell">
    <aside class="rail">
      <header class="brand"><div class="orb"><i></i></div><div><span class="eyebrow">TRAINING OBSERVATORY</span><h1>Luna Pilot</h1></div></header>
      <div class="runline"><span class="live-dot"></span><span id="runName"></span></div>
      <div class="stats" id="stats"></div>
      <label class="search"><span>⌕</span><input id="search" placeholder="搜索题目、答案、ID…"></label>
      <div class="filters"><button class="filter active" data-filter="all">全部</button><button class="filter" data-filter="5">得分 5</button><button class="filter" data-filter="0">得分 0</button></div>
      <div class="rollout-list" id="rolloutList"></div>
    </aside>
    <main class="stage" id="stage">
      <div class="topbar"><div><span class="crumb">GPT-5.6-LUNA / PILOT /</span> <strong id="currentId"></strong></div><div class="key"><span><b class="swatch model"></b>模型</span><span><b class="swatch tool"></b>工具</span><span><b class="swatch media"></b>媒体</span></div></div>
      <section id="content"></section>
    </main>
  </div>
  <dialog id="lightbox"><button id="closeLightbox" aria-label="关闭">×</button><img id="lightboxImage"><p id="lightboxCaption"></p></dialog>
  <script src="data.js"></script><script src="app.js"></script>
</body>
</html>'''


STYLES = r''':root{--bg:#080b10;--panel:#10141b;--line:#252c38;--text:#eef2f6;--muted:#8d98a8;--blue:#61a8ff;--cyan:#64e1d1;--amber:#f8bd66;--red:#ff7b76;--green:#55d68b;--mono:"SFMono-Regular",Consolas,"Liberation Mono",monospace;--sans:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--text);font-family:var(--sans)}body{overflow:hidden}.shell{height:100vh;display:grid;grid-template-columns:344px 1fr}.rail{min-width:0;border-right:1px solid var(--line);background:linear-gradient(180deg,#10141b,#0b0f15);display:flex;flex-direction:column}.brand{display:flex;align-items:center;gap:14px;padding:23px 21px 12px}.brand h1{font-size:22px;margin:2px 0 0;letter-spacing:-.03em}.eyebrow{font:10px var(--mono);letter-spacing:.15em;color:var(--muted)}.orb{position:relative;width:35px;height:35px;border:1px solid #568ed6;border-radius:50%;display:grid;place-items:center;box-shadow:0 0 18px #478cd533}.orb:before,.orb:after,.orb i{content:"";position:absolute;width:17px;height:17px;border:1px solid #70beff;border-radius:50%;transform:translate(5px)}.orb:after{transform:translate(-5px)}.orb i{transform:none;border-color:#71f0de}.runline{margin:0 21px 16px;padding:9px 11px;border:1px solid var(--line);background:#0a0e14;border-radius:7px;font:11px var(--mono);color:#aab4c2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.live-dot{display:inline-block;width:6px;height:6px;margin-right:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}.stats{display:grid;grid-template-columns:repeat(3,1fr);margin:0 21px 16px;border:1px solid var(--line);border-radius:8px;overflow:hidden}.stat{padding:10px;border-right:1px solid var(--line)}.stat:last-child{border:0}.stat b{display:block;font:18px var(--mono)}.stat small{display:block;margin-top:3px;font:9px var(--mono);letter-spacing:.08em;color:var(--muted)}.search{margin:0 21px 10px;height:38px;border:1px solid var(--line);border-radius:7px;background:#080b10;display:flex;align-items:center;gap:8px;padding:0 10px;color:var(--muted)}.search input{width:100%;color:var(--text);border:0;outline:0;background:transparent;font:12px var(--sans)}.filters{display:flex;gap:6px;margin:0 21px 14px}.filter{border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--muted);font:10px var(--mono);padding:6px 9px;cursor:pointer}.filter.active{color:var(--text);border-color:#4c75a5;background:#17304f}.rollout-list{overflow:auto;padding:0 12px 24px}.rollout-item{width:100%;text-align:left;color:var(--text);border:1px solid transparent;background:transparent;border-radius:7px;padding:10px;margin:0 0 4px;cursor:pointer;display:grid;grid-template-columns:38px 1fr auto;gap:9px;align-items:start}.rollout-item:hover{background:#161b23}.rollout-item.active{background:#172131;border-color:#29496e}.rollout-item .num{font:10px var(--mono);color:var(--muted);padding-top:2px}.rollout-item .q{font-size:12px;line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.badge{display:inline-flex;align-items:center;justify-content:center;min-width:25px;height:20px;border-radius:4px;font:bold 10px var(--mono)}.badge.pass{background:#153828;color:#72e2a5}.badge.fail{background:#3c2021;color:#ff9690}.stage{height:100vh;overflow:auto;background:radial-gradient(circle at 70% -10%,#17263a,transparent 34%),var(--bg)}.topbar{height:59px;position:sticky;z-index:5;top:0;display:flex;justify-content:space-between;align-items:center;padding:0 28px;border-bottom:1px solid var(--line);background:#080b10e8;backdrop-filter:blur(14px);font:11px var(--mono)}.crumb{color:var(--muted)}.key{display:flex;gap:16px;color:var(--muted)}.swatch{display:inline-block;width:7px;height:7px;border-radius:2px;margin-right:5px}.swatch.model{background:var(--blue)}.swatch.tool{background:var(--amber)}.swatch.media{background:var(--cyan)}#content{max-width:1320px;margin:auto;padding:28px}.hero{display:grid;grid-template-columns:1fr 160px;gap:24px;padding-bottom:26px;border-bottom:1px solid var(--line)}.hero h2{font-size:25px;line-height:1.34;letter-spacing:-.025em;margin:8px 0 0}.label{font:10px var(--mono);letter-spacing:.13em;color:var(--muted)}.score-card{border:1px solid var(--line);border-radius:10px;background:linear-gradient(145deg,#151b24,#0c1016);display:grid;place-items:center;text-align:center}.score-card b{font:47px var(--mono);line-height:1}.score-card.pass b{color:var(--green)}.score-card.fail b{color:var(--red)}.score-card span{display:block;margin-top:8px;font:9px var(--mono);letter-spacing:.12em;color:var(--muted)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:22px 0}.answer-card{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:17px 18px}.answer-card.answer{border-top:2px solid var(--blue)}.answer-card.reference{border-top:2px solid var(--cyan)}.answer-card p,.reflection p{font-size:13px;line-height:1.65;color:#ccd3dd;margin:9px 0 0;white-space:pre-wrap}.reflection{padding:17px 18px;border-left:2px solid #445264;background:#0d1118;margin:0 0 16px}.reflection.meta{border-color:#755db4}.section-head{display:flex;justify-content:space-between;align-items:end;margin:30px 0 14px}.section-head h3{font-size:18px;margin:3px 0 0}.section-head .count{font:10px var(--mono);color:var(--muted)}.attempt{border-left:1px solid var(--line);margin-left:9px;padding-left:22px}.attempt-title{display:flex;align-items:center;gap:10px;margin:18px 0;font:11px var(--mono);color:var(--muted)}.attempt-title:before{content:"";width:7px;height:7px;margin-left:-26px;border-radius:50%;background:#506075;border:3px solid var(--bg)}.error-box{border:1px solid #5c3030;background:#241517;border-radius:7px;padding:12px;margin-bottom:14px;color:#ffaaa6;font:11px/1.6 var(--mono);white-space:pre-wrap}.trace{border:1px solid var(--line);border-radius:9px;background:var(--panel);margin:0 0 13px;overflow:hidden}.trace-head{display:flex;align-items:center;gap:9px;padding:10px 13px;border-bottom:1px solid var(--line);background:#121821}.trace-head .seg{font:9px var(--mono);color:#8cbfff;border:1px solid #2e527d;border-radius:4px;padding:4px 6px}.trace-head strong{font:12px var(--mono);font-weight:600}.trace-head .file{margin-left:auto;color:var(--muted);font:9px var(--mono)}.trace-input{padding:12px 14px;border-bottom:1px dashed var(--line);font:11px/1.55 var(--mono);color:#aab4c2;white-space:pre-wrap}.events{padding:10px}.event{display:grid;grid-template-columns:88px 1fr;gap:12px;margin:6px 0;padding:9px 10px;border-radius:6px;background:#0b0f15;border-left:2px solid #48566a}.event.model{border-color:var(--blue)}.event.call{border-color:var(--amber)}.event.output{border-color:var(--cyan)}.event .etype{font:9px/1.4 var(--mono);letter-spacing:.08em;color:var(--muted);text-transform:uppercase}.event pre{margin:0;white-space:pre-wrap;word-break:break-word;font:10.5px/1.55 var(--mono);color:#c7ced8;max-height:360px;overflow:auto}.reasoning-note{color:#728096!important;font-style:italic}.trace-final{margin:0 10px 10px;border:1px solid #29496e;background:#0e1a29;border-radius:6px;padding:10px;font:10.5px/1.55 var(--mono);white-space:pre-wrap}.media-strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:9px;padding:0 10px 10px}.media-card{position:relative;min-width:0;border:1px solid var(--line);border-radius:7px;overflow:hidden;background:#080b10;cursor:zoom-in}.media-card img{width:100%;height:118px;object-fit:cover;display:block}.media-card .caption{padding:7px 8px;font:8.5px/1.35 var(--mono);color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.media-card .kind{position:absolute;top:6px;left:6px;padding:3px 5px;border-radius:3px;background:#071118d9;color:var(--cyan);font:8px var(--mono);text-transform:uppercase}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:11px}.gallery .media-card img{height:145px}.unlinked{opacity:.72}details.extra-media{margin-top:13px}details.extra-media summary{cursor:pointer;color:var(--muted);font:11px var(--mono)}dialog{width:min(94vw,1500px);border:1px solid #344052;border-radius:10px;padding:0;background:#090d13;color:white;box-shadow:0 30px 100px #000}dialog::backdrop{background:#020408dd}dialog img{display:block;max-width:100%;max-height:82vh;margin:auto}dialog p{padding:9px 14px;margin:0;border-top:1px solid var(--line);font:10px var(--mono);color:var(--muted);word-break:break-all}dialog button{position:fixed;right:3vw;top:2vh;width:34px;height:34px;border:1px solid #ffffff55;border-radius:50%;background:#05070acc;color:white;font-size:24px;cursor:pointer}@media(max-width:900px){body{overflow:auto}.shell{height:auto;display:block}.rail{height:auto}.rollout-list{max-height:310px}.stage{height:auto;overflow:visible}.topbar{top:0}.hero{grid-template-columns:1fr}.score-card{height:100px}.grid2{grid-template-columns:1fr}.key{display:none}#content{padding:20px}.event{grid-template-columns:1fr}.shell{min-height:100vh}}'''


APP_JS = r'''const data=window.TRACE_DATA;let selected=null;let scoreFilter="all";const el=id=>document.getElementById(id);const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));
function questionFor(r){const traces=r.attempts.flatMap(a=>a.traces),answer=traces.find(t=>t.kind==="answer");return (answer?.input||"").replace(/^Question:\s*/i,"").split("\n\nAvailable skills:")[0]||`Question ${r.question_index}`}
function mediaCard(m){return `<article class="media-card ${m.linked?"":"unlinked"}" data-image="${esc(m.url)}" data-caption="${esc(m.source_name)}"><span class="kind">${m.kind}</span><img loading="lazy" src="${esc(m.url)}" alt="${esc(m.source_name)}"><div class="caption">${m.timestamp_s==null?"":formatTime(m.timestamp_s)+" · "}${esc(m.source_name)}</div></article>`}
function formatTime(seconds){const h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60),s=(seconds%60).toFixed(1);return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(4,"0")}`}
function pretty(s){if(typeof s!=="string")return JSON.stringify(s,null,2);try{return JSON.stringify(JSON.parse(s),null,2)}catch{return s}}
function eventText(e){if(e.type==="reasoning")return e.content?.length?JSON.stringify(e.content,null,2):"Reasoning item preserved in trace; encrypted payload omitted from published site.";if(e.type==="function_call")return `${e.name}(${pretty(e.arguments)})\nstatus: ${e.status??"—"}\ncall_id: ${e.call_id??"—"}`;if(e.type==="function_call_output")return typeof e.output==="string"?pretty(e.output):JSON.stringify(e.output,null,2);return typeof e.content==="string"?e.content:JSON.stringify(e,null,2)}
function eventClass(e){return e.type==="function_call"?"call":e.type==="function_call_output"?"output":"model"}
function eventLabel(e){return e.type==="function_call"?`TOOL · ${e.name}`:e.type==="function_call_output"?"TOOL RETURN":e.type==="reasoning"?"MODEL · REASONING":`MODEL · ${e.role||e.type||"MESSAGE"}`}
function renderList(){const query=el("search").value.trim().toLowerCase();const shown=data.rollouts.filter(r=>(scoreFilter==="all"||String(r.score)===scoreFilter)&&(!query||`${r.id} ${questionFor(r)} ${r.answer} ${r.reference_answer}`.toLowerCase().includes(query)));el("rolloutList").innerHTML=shown.map((r,i)=>`<button class="rollout-item ${selected?.id===r.id?"active":""}" data-id="${esc(r.id)}"><span class="num">${String(i+1).padStart(2,"0")}</span><span class="q">Q${r.question_index} · R${r.rollout}<br>${esc(questionFor(r))}</span><span class="badge ${r.score?"pass":"fail"}">${r.score}</span></button>`).join("")||'<p style="padding:16px;color:var(--muted);font-size:12px">没有匹配项</p>';document.querySelectorAll(".rollout-item").forEach(b=>b.onclick=()=>selectRollout(b.dataset.id))}
function traceHtml(t,a,r){const media=t.media_refs.map(name=>r.media.find(m=>m.attempt===a.name&&m.source_name===name)).filter(Boolean);return `<article class="trace"><div class="trace-head"><span class="seg">SEG ${String(t.segment).padStart(2,"0")}</span><strong>${esc(t.kind)}</strong><span class="file">${esc(t.file)}</span></div>${t.input?`<div class="trace-input"><span class="label">SEGMENT INPUT</span>\n${esc(t.input)}</div>`:""}<div class="events">${t.history.map(e=>`<div class="event ${eventClass(e)}"><div class="etype">${esc(eventLabel(e))}</div><pre class="${e.type==="reasoning"?"reasoning-note":""}">${esc(eventText(e))}</pre></div>`).join("")}</div>${t.final_output!=null?`<div class="trace-final"><span class="label">FINAL OUTPUT</span>\n${esc(pretty(t.final_output))}</div>`:""}${media.length?`<div class="media-strip">${media.map(mediaCard).join("")}</div>`:""}</article>`}
function selectRollout(id){selected=data.rollouts.find(r=>r.id===id);history.replaceState(null,"",`#${encodeURIComponent(id)}`);renderList();el("currentId").textContent=selected.id;const attempts=selected.attempts.map(a=>`<div class="attempt"><div class="attempt-title">${esc(a.name)} · ${a.traces.length} segments${a.error?" · ERROR":""}</div>${a.error?`<div class="error-box">${esc(`${a.error.type}: ${a.error.message}`)}</div>`:""}${a.traces.map(t=>traceHtml(t,a,selected)).join("")}</div>`).join("");const linked=new Set(selected.attempts.flatMap(a=>a.traces.flatMap(t=>t.media_refs.map(name=>`${a.name}/${name}`)))),extra=selected.media.filter(m=>!linked.has(`${m.attempt}/${m.source_name}`));el("content").innerHTML=`<div class="hero"><div><span class="label">QUESTION · Q${selected.question_index} · ROLLOUT ${selected.rollout}</span><h2>${esc(questionFor(selected))}</h2></div><div class="score-card ${selected.score?"pass":"fail"}"><div><b>${selected.score}</b><span>REWARD SCORE</span></div></div></div><div class="grid2"><article class="answer-card answer"><span class="label">MODEL ANSWER</span><p>${esc(selected.answer)}</p></article><article class="answer-card reference"><span class="label">REFERENCE</span><p>${esc(selected.reference_answer)}</p></article></div><div class="reflection"><span class="label">REFLECTION</span><p>${esc(selected.reflection)}</p></div>${selected.meta_reflection?`<div class="reflection meta"><span class="label">META REFLECTION</span><p>${esc(selected.meta_reflection)}</p></div>`:""}<div class="section-head"><div><span class="label">EXECUTION TRACE</span><h3>模型分段与工具往返</h3></div><span class="count">${selected.attempts.reduce((n,a)=>n+a.traces.length,0)} SEGMENTS · ${selected.media.length} MEDIA</span></div>${attempts}${extra.length?`<details class="extra-media"><summary>展开 ${extra.length} 个未在可见返回文本中点名的生成媒体</summary><div class="gallery" style="margin-top:12px">${extra.map(mediaCard).join("")}</div></details>`:""}`;wireImages();el("stage").scrollTop=0}
function wireImages(){document.querySelectorAll(".media-card").forEach(c=>c.onclick=()=>{el("lightboxImage").src=c.dataset.image;el("lightboxCaption").textContent=c.dataset.caption;el("lightbox").showModal()})}
el("runName").textContent=data.run;el("stats").innerHTML=`<div class="stat"><b>${data.rollout_count}</b><small>ROLLOUTS</small></div><div class="stat"><b>${data.trace_count}</b><small>SEGMENTS</small></div><div class="stat"><b>${data.media_count}</b><small>MEDIA</small></div>`;el("search").oninput=renderList;document.querySelectorAll(".filter").forEach(b=>b.onclick=()=>{document.querySelectorAll(".filter").forEach(x=>x.classList.remove("active"));b.classList.add("active");scoreFilter=b.dataset.filter;renderList()});el("closeLightbox").onclick=()=>el("lightbox").close();el("lightbox").onclick=e=>{if(e.target===el("lightbox"))el("lightbox").close()};renderList();const hash=decodeURIComponent(location.hash.slice(1));selectRollout(data.rollouts.some(r=>r.id===hash)?hash:data.rollouts[0].id);'''


if __name__ == "__main__":
    main()
