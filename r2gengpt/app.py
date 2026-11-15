"""A local web page: choose a chest X-ray, get a generated report. Example:
    python app.py                                   # then open http://127.0.0.1:7860
    python app.py --checkpoint runs/iu_delta_tinyllama/best.pt --port 8000

Everything runs on this machine; images are never sent anywhere else.
"""
import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

from r2gen.builder import generation_kwargs, get_device, load_checkpoint
from r2gen.data import build_transform, clean_indication
from r2gen.labeler import CONDITIONS

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "runs", "iu_delta_tinyllama_disease_llmlora", "last.pt")

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>X-ray Report</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#5d6678; --line:#dfe3ea; --accent:#2457c5; --warn:#8a5a00; }
  @media (prefers-color-scheme: dark) { :root { --bg:#14171c; --card:#1d2128; --text:#e6e9ef; --muted:#9aa3b2;
    --line:#2e3440; --accent:#7aa2ff; --warn:#e0b257; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 -apple-system, system-ui, sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 32px 16px 48px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .slots { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
  label.slot { display: flex; flex-direction: column; gap: 8px; cursor: pointer; }
  .drop { border: 2px dashed var(--line); border-radius: 10px; aspect-ratio: 1; display: grid; place-items: center;
          overflow: hidden; color: var(--muted); text-align: center; padding: 8px; }
  .drop img { width: 100%; height: 100%; object-fit: contain; background: #000; }
  .drop.over { border-color: var(--accent); }
  input[type=file] { display: none; }
  input[type=text] { width: 100%; padding: 10px; border: 1px solid var(--line); border-radius: 8px;
                     background: var(--bg); color: var(--text); font: inherit; }
  button { background: var(--accent); color: #fff; border: 0; border-radius: 8px; padding: 12px 20px; font: inherit;
           font-weight: 600; cursor: pointer; margin-top: 16px; }
  button:disabled { opacity: .5; cursor: default; }
  .report { font-size: 17px; white-space: pre-wrap; }
  .tags span { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 10px; margin: 4px 6px 0 0; }
  .muted { color: var(--muted); font-size: 14px; }
  .warn { color: var(--warn); font-size: 14px; }
</style></head>
<body><main>
  <h1>Chest X-ray report</h1>
  <p class="sub">Choose a frontal chest X-ray (and optionally the lateral view), then generate a report.</p>
  <div class="card">
    <div class="slots">
      <label class="slot"><b>Frontal view (required)</b><div class="drop" id="d0">Click or drop an image</div>
        <input type="file" id="f0" accept="image/*"></label>
      <label class="slot"><b>Lateral view (optional)</b><div class="drop" id="d1">Click or drop an image</div>
        <input type="file" id="f1" accept="image/*"></label>
    </div>
    <p style="margin:16px 0 4px"><b>Clinical indication</b> <span class="muted">(optional)</span></p>
    <input type="text" id="ind" placeholder="e.g. chest pain, shortness of breath">
    <button id="go" disabled>Generate report</button>
  </div>
  <div class="card" id="out" hidden>
    <b>Generated report</b>
    <p class="report" id="report"></p>
    <div id="findings"></div>
    <p class="muted" id="timing"></p>
  </div>
  <p class="warn">Research model (TinyLlama-1.1B, trained on ~2,000 IU-Xray studies). It misses many abnormal findings
  and can invent some. Not for clinical use.</p>
  <p class="muted">Model: <code>__CHECKPOINT__</code></p>
</main>
<script>
const images = [null, null];
function load(i, file) {
  if (!file) return;
  const r = new FileReader();
  r.onload = () => { images[i] = r.result;
    document.getElementById('d' + i).innerHTML = '<img alt="view ' + (i + 1) + '" src="' + r.result + '">';
    document.getElementById('go').disabled = !images[0]; };
  r.readAsDataURL(file);
}
[0, 1].forEach(i => {
  const input = document.getElementById('f' + i), drop = document.getElementById('d' + i);
  input.addEventListener('change', () => load(i, input.files[0]));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); load(i, e.dataTransfer.files[0]); });
});
document.getElementById('go').addEventListener('click', async () => {
  const btn = document.getElementById('go');
  btn.disabled = true; btn.textContent = 'Generating...';
  const t0 = performance.now();
  try {
    const res = await fetch('/report', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: images.filter(Boolean), indication: document.getElementById('ind').value }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    document.getElementById('report').textContent = data.report;
    document.getElementById('findings').innerHTML = data.findings === null ? '' :
      '<b>Predicted findings</b><div class="tags">' + (data.findings.length ? data.findings.map(f =>
        '<span>' + f.name + ' (' + Math.round(f.probability * 100) + '%)</span>').join('') : '<span>none</span>') + '</div>';
    document.getElementById('timing').textContent = 'Took ' + ((performance.now() - t0) / 1000).toFixed(1) + ' s';
    document.getElementById('out').hidden = false;
  } catch (err) {
    document.getElementById('report').textContent = 'Error: ' + err.message;
    document.getElementById('findings').innerHTML = ''; document.getElementById('out').hidden = false;
  } finally { btn.disabled = false; btn.textContent = 'Generate report'; }
});
</script></body></html>"""


class ReportService:
    def __init__(self, checkpoint):
        self.device = get_device()
        self.model, self.args = load_checkpoint(checkpoint, self.device)
        self.transform = build_transform(self.args.image_size, mean=self.args.image_mean, std=self.args.image_std)
        self.lock = threading.Lock()  # one generation at a time on the GPU

    def report(self, data_urls, indication=""):
        views = []
        for url in data_urls[: self.args.max_views]:
            raw = base64.b64decode(url.split(",", 1)[-1])
            views.append(self.transform(Image.open(io.BytesIO(raw)).convert("RGB")))
        while len(views) < self.args.max_views:
            views.append(views[-1])  # one view only: repeat it, as during training
        images = torch.stack(views).unsqueeze(0).to(self.device)
        with self.lock:
            reports, probs = self.model.generate(images, indications=[clean_indication(indication)],
                                                 return_findings=True, **generation_kwargs(self.args))
        findings = None
        if probs is not None:
            findings = [{"name": c, "probability": round(p, 3)} for c, p in zip(CONDITIONS, probs[0].tolist())
                        if p > 0.5]
        return {"report": reports[0], "findings": findings}


def make_handler(service, page):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/report":
                return self._send(404, "not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not body.get("images"):
                    raise ValueError("no image received")
                result = service.report(body["images"], body.get("indication", ""))
                self._send(200, json.dumps(result), "application/json")
            except Exception as err:  # bad image, decoding error, ...
                self._send(400, json.dumps({"error": str(err)}), "application/json")

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps the page private to this machine.")
    p.add_argument("--port", type=int, default=7860)
    cli = p.parse_args()

    print(f"loading {cli.checkpoint} ...")
    service = ReportService(cli.checkpoint)
    page = PAGE.replace("__CHECKPOINT__", os.path.relpath(cli.checkpoint, os.path.dirname(os.path.abspath(__file__))))
    server = ThreadingHTTPServer((cli.host, cli.port), make_handler(service, page))
    print(f"ready: open http://{cli.host}:{cli.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
