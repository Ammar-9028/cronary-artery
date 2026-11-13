(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  // ---- animated ECG background ----
  var cv = $('#ecg'), cx = cv.getContext('2d'), W, H, x = 0, pts = [];
  function rs() { W = cv.width = innerWidth; H = cv.height = innerHeight; pts = []; x = 0; }
  rs(); addEventListener('resize', rs);
  function wave(t) { var p = t % 160;
    if (p < 20) return 0; if (p < 30) return -8; if (p < 38) return 0; if (p < 46) return 6;
    if (p < 50) return -70; if (p < 54) return 40; if (p < 60) return 0; if (p < 80) return -14 * Math.sin((p - 60) / 20 * Math.PI); return 0; }
  (function draw() {
    cx.fillStyle = 'rgba(7,11,26,.08)'; cx.fillRect(0, 0, W, H);
    x += 3; if (x > W) { x = 0; cx.clearRect(0, 0, W, H); }
    var y = H * 0.72 + wave(x);
    cx.strokeStyle = '#ff3b5c'; cx.lineWidth = 2; cx.shadowColor = '#ff3b5c'; cx.shadowBlur = 12;
    if (pts.length) { cx.beginPath(); cx.moveTo(pts[0], pts[1]); cx.lineTo(x, y); cx.stroke(); }
    pts = [x, y];
    if (!matchMedia('(prefers-reduced-motion:reduce)').matches) requestAnimationFrame(draw);
  })();
  // ---- count-up stats ----
  document.querySelectorAll('[data-n]').forEach(function (el) {
    var n = +el.dataset.n, i = 0, step = Math.max(1, Math.ceil(n / 40));
    var t = setInterval(function () { i = Math.min(n, i + step); el.textContent = i; if (i >= n) clearInterval(t); }, 25);
  });
  // ---- upload ----
  var drop = $('#drop'), file = $('#file'), go = $('#go'), err = $('#err'), cur = null;
  function setFile(f) {
    err.textContent = '';
    if (!f) return;
    if (!/^image\//.test(f.type)) { err.textContent = 'Sirf image file chuno.'; return; }
    if (f.size > 8 * 1024 * 1024) { err.textContent = 'File 8 MB se badi hai.'; return; }
    cur = f; $('#prev').src = URL.createObjectURL(f);
    $('.prev').hidden = false; $('.ph').hidden = true; go.disabled = false;
  }
  drop.addEventListener('click', function () { file.click(); });
  drop.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } });
  file.addEventListener('change', function () { setFile(file.files[0]); });
  ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (e) { setFile(e.dataTransfer.files[0]); });
  // ---- analyze ----
  var data = null;
  go.addEventListener('click', function () {
    if (!cur) return;
    err.textContent = ''; go.disabled = true; go.textContent = 'Analyzing...'; drop.classList.add('busy');
    var fd = new FormData(); fd.append('image', cur);
    var started = Date.now();
    fetch('/predict', { method: 'POST', body: fd })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || 'Server error'); return j; }); })
      .then(function (j) { var wait = Math.max(0, 1800 - (Date.now() - started)); setTimeout(function () { show(j); }, wait); })
      .catch(function (e) { err.textContent = e.message; reset(); });
  });
  function reset() { go.disabled = false; go.textContent = 'Analyze karo'; drop.classList.remove('busy'); }
  function show(j) {
    data = j; reset();
    $('#res').classList.remove('idle'); $('.out').hidden = false;
    var pos = j.positive, arc = $('#arc'), v = $('#verdict');
    v.textContent = 'CAD ' + j.label; v.className = pos ? 'pos' : 'neg';
    arc.style.stroke = pos ? '#ff3b5c' : '#34d399'; arc.style.color = arc.style.stroke;
    requestAnimationFrame(function () { arc.style.strokeDashoffset = 327 * (1 - j.probability); });
    var p = 0, target = Math.round(j.probability * 100), t = setInterval(function () { p = Math.min(target, p + 2); $('#pct').textContent = p; if (p >= target) clearInterval(t); }, 25);
    $('#conf').textContent = 'Confidence ' + (j.confidence * 100).toFixed(1) + '%  |  threshold ' + j.threshold.toFixed(2);
    $('#meta').textContent = (j.demo ? 'DEMO result (model not loaded) | ' : '') + j.ms + ' ms';
    $('#heatTab').disabled = !j.heatmap; tab('original');
  }
  function tab(v) {
    document.querySelectorAll('.tabs button').forEach(function (b) { b.classList.toggle('on', b.dataset.v === v); });
    $('#view').src = (v === 'heat' && data.heatmap) ? data.heatmap : data.original;
  }
  document.querySelectorAll('.tabs button').forEach(function (b) { b.addEventListener('click', function () { if (data && !b.disabled) tab(b.dataset.v); }); });
})();
