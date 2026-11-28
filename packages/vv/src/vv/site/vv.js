/* Shared helpers for verification and validation sites built on vv.
 *
 * A project's page loads vv.css and this file, then builds its own view from
 * `window.vv`: formatting, tooltips, the published records it fetches, Plotly
 * set up for light and dark, and `vv.explorer`, a filterable, sortable table
 * of benchmarks with an expandable detail row. What a benchmark is, which
 * columns it has and what its detail shows stay in the project's page.
 */
"use strict";
(function () {
  var vv = {};

  /* ---------- small helpers ---------- */

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function num(v) { return (typeof v === "number" && isFinite(v)) ? v : null; }
  function fmtInt(n) {
    if (n >= 10000) {
      var v = n >= 1e6 ? n / 1e6 : n / 1e3;
      return (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + (n >= 1e6 ? "M" : "K");
    }
    return n.toLocaleString("en-US");
  }
  function fmtPct(v) {
    if (v === null || v === undefined || !isFinite(v)) return "n/a";
    if (v >= 100) return v.toFixed(0) + "%";
    return (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "%";
  }
  /* A value to `digits` significant figures, in exponent form when very large or small. */
  function fmtNum(v, digits) {
    if (num(v) === null) return "n/a";
    digits = digits || 4;
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(digits - 1);
    return Number(v.toPrecision(digits)).toString();
  }
  function fmtDuration(seconds) {
    if (seconds >= 365.25 * 86400) return (seconds / (365.25 * 86400)).toPrecision(3) + " y";
    if (seconds >= 86400) return (seconds / 86400).toPrecision(3) + " d";
    if (seconds >= 3600) return (seconds / 3600).toPrecision(3) + " h";
    if (seconds >= 60) return (seconds / 60).toPrecision(3) + " min";
    return seconds.toPrecision(3) + " s";
  }
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    var pos = (sorted.length - 1) * q;
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  /* Nuclides sort by (element, mass number), Mo92 before Mo100; anything else as text. */
  var NAT_RE = /^([A-Za-z]+)(\d+)?$/;
  function natCompare(a, b) {
    a = String(a); b = String(b);
    var ma = NAT_RE.exec(a), mb = NAT_RE.exec(b);
    if (!ma || !mb) return a < b ? -1 : a > b ? 1 : 0;
    if (ma[1] !== mb[1]) return ma[1] < mb[1] ? -1 : 1;
    return (ma[2] ? parseInt(ma[2], 10) : -1) - (mb[2] ? parseInt(mb[2], 10) : -1);
  }
  function tile(label, value, note) {
    var t = el("div", "card tile");
    t.appendChild(el("div", "label", label));
    t.appendChild(el("div", "value", value));
    if (note) t.appendChild(el("div", "note", note));
    return t;
  }
  function domTable(headers, rows, labelCols) {
    if (labelCols === undefined) labelCols = 1;
    var wrap = el("div", "tally-wrap");
    var table = el("table", "tally");
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    headers.forEach(function (h) {
      var th = el("th", null, h);
      th.scope = "col";
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    rows.forEach(function (cells) {
      var tr = document.createElement("tr");
      cells.forEach(function (c, i) {
        tr.appendChild(el("td", i >= labelCols ? "val" : null, c));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  /* ---------- tooltip ---------- */

  function tooltipBox() {
    var t = document.getElementById("vv-tooltip");
    if (!t) {
      t = el("div", "tooltip");
      t.id = "vv-tooltip";
      t.setAttribute("role", "tooltip");
      document.body.appendChild(t);
    }
    return t;
  }
  function hideTip() { var t = document.getElementById("vv-tooltip"); if (t) t.style.display = "none"; }
  /* `content()` returns {strong, rest}: a bold first line and the text under it. */
  function attachTip(target, content) {
    function show(x, y) {
      var t = tooltipBox();
      t.textContent = "";
      var c = content();
      t.appendChild(el("strong", null, c.strong));
      if (c.rest) {
        t.appendChild(document.createElement("br"));
        t.appendChild(document.createTextNode(c.rest));
      }
      t.style.display = "block";
      var pad = 12;
      var w = t.offsetWidth, h = t.offsetHeight;
      var left = Math.min(x + pad, window.innerWidth - w - 8);
      var top = y - h - pad;
      if (top < 8) top = y + pad;
      t.style.left = left + "px";
      t.style.top = top + "px";
    }
    target.addEventListener("pointermove", function (ev) { show(ev.clientX, ev.clientY); });
    target.addEventListener("pointerleave", hideTip);
    target.addEventListener("focus", function () {
      var r = target.getBoundingClientRect();
      show(r.left + r.width / 2, r.top);
    });
    target.addEventListener("blur", hideTip);
  }

  /* ---------- published records ---------- */

  /* Undo vv.records' encoding of arrays ({"_np_array": [...]}) */
  function decodeNp(o) {
    if (o && typeof o === "object") {
      if (o._np_array !== undefined) return o._np_array;
      if (Array.isArray(o)) return o.map(decodeNp);
      var out = {};
      for (var k in o) out[k] = decodeNp(o[k]);
      return out;
    }
    return o;
  }
  var recordStore = {};
  /* One published record (`records/<blob>.json`), fetched once:
     {identity, settings, results}, with the results decoded. */
  function fetchRecord(path) {
    if (!recordStore[path]) {
      recordStore[path] = fetch(path)
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status + " for " + path);
          return r.json();
        })
        .then(function (d) {
          return { identity: d.code_identity || {}, settings: d.settings || {}, results: decodeNp(d.results || {}) };
        });
      recordStore[path].catch(function () { delete recordStore[path]; });
    }
    return recordStore[path];
  }

  /* Fetch the site's index; on failure say so in `errorBox` and how to preview locally. */
  function loadIndex(url, errorBox, previewHint) {
    return fetch(url)
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .catch(function (err) {
        errorBox.hidden = false;
        errorBox.appendChild(el("p", null, "Could not load " + url + " (" + err.message + ")."));
        var p = el("p", null, "The index is built by CI next to this page. ");
        if (previewHint) p.appendChild(document.createTextNode(previewHint + " "));
        p.appendChild(document.createTextNode("fetch does not work from file://, so serve the site over HTTP."));
        errorBox.appendChild(p);
        throw err;
      });
  }

  /* "Run <tag>, published <date>, with <codes>." and the run it is compared with. */
  function runLine(node, run, previous) {
    node.textContent = "";
    node.appendChild(document.createTextNode("Run "));
    node.appendChild(el("code", null, run.tag));
    node.appendChild(document.createTextNode(", published " + new Date(run.date).toUTCString()
      + ", with " + run.codes.join(", ") + "."));
    if (previous) {
      node.appendChild(document.createTextNode(" Compared with the run before, "));
      node.appendChild(el("code", null, previous.tag));
      node.appendChild(document.createTextNode(" (" + previous.codes.join(", ") + ")."));
    }
    node.hidden = false;
  }

  /* ---------- Plotly, light and dark ---------- */

  var PLOTLY_SRC = "https://cdn.plot.ly/plotly-2.35.2.min.js";
  var plotlyPromise = null;
  function ensurePlotly() {
    if (window.Plotly) return Promise.resolve();
    if (!plotlyPromise) {
      plotlyPromise = new Promise(function (resolve, reject) {
        var s = document.createElement("script");
        s.src = PLOTLY_SRC;
        s.onload = resolve;
        s.onerror = function () {
          plotlyPromise = null;
          reject(new Error("could not load plotly.js from " + PLOTLY_SRC));
        };
        document.head.appendChild(s);
      });
    }
    return plotlyPromise;
  }
  function isDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function plotlyLayout(extra) {
    var dark = isDark();
    var base = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: dark ? "#c3c2b7" : "#52514e", size: 11 },
      margin: { l: 60, r: 16, t: 34, b: 42 },
      height: 460
    };
    for (var k in extra) base[k] = extra[k];
    return base;
  }
  function axisDefaults(extra) {
    var dark = isDark();
    var base = { gridcolor: dark ? "#2c2c2a" : "#e1e0d9",
                 linecolor: dark ? "#383835" : "#c3c2b7", zeroline: false };
    for (var k in extra) base[k] = extra[k];
    return base;
  }
  /* Histogram-style steps through bin edges `bins` (one more than `values`). */
  function stepXY(bins, values) {
    var x = [], y = [];
    for (var i = 0; i < values.length; i++) {
      x.push(bins[i], bins[i + 1]);
      y.push(values[i], values[i]);
    }
    return { x: x, y: y };
  }
  /* A series palette that reads in light and dark. */
  var SERIES_COLORS = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4", "#eb6834", "#008300"];

  /* ---------- the benchmark explorer ---------- */

  /* A filter row, a sortable table and an expandable detail row, in `config.mount`.
   *
   * config.records   [{id, ...}]: one per benchmark row
   * config.filters   [{key, label, format(value)}]: a select per field, with counts
   * config.toggles   [{key, label, test(rec)}]: checkboxes keeping rows that pass
   * config.columns   [{key, label, value(rec), render(td, rec), sortable, sortValue(rec), firstDir, cls}]
   * config.sort      {key, dir}: the default order (dir 1 ascending, -1 descending)
   * config.detail    function (box, rec): fills an opened row (optional)
   * config.summary   function (recs): called with the shown rows on every render (optional)
   * config.noun      what a row is, for the count ("benchmarks")
   * config.caption   the table's accessible caption
   */
  function explorer(config) {
    var filters = {}, toggles = {};
    var defaultSort = config.sort || { key: config.columns[0].key, dir: 1 };
    var sort = { key: defaultSort.key, dir: defaultSort.dir };
    var open = null;
    var columns = config.columns.slice();
    var uid = "vvx" + Math.random().toString(36).slice(2, 8);
    (config.filters || []).forEach(function (f) { filters[f.key] = ""; });
    (config.toggles || []).forEach(function (t) { toggles[t.key] = false; });
    if (config.detail) {
      columns.unshift({ key: "_expand", label: "", render: function (td, r) {
        var b = el("button", "expander", "▸");
        b.type = "button";
        b.setAttribute("aria-expanded", "false");
        b.setAttribute("aria-label", "Show details for " + r.id);
        td.appendChild(b);
      } });
    }

    var mount = config.mount;
    var bar = el("div", "filters");
    var selects = {};
    (config.filters || []).forEach(function (f) {
      var field = el("div", "field");
      var label = el("label", null, f.label);
      label.htmlFor = uid + "-" + f.key;
      var sel = document.createElement("select");
      sel.id = label.htmlFor;
      sel.addEventListener("change", function () { filters[f.key] = sel.value; render(); });
      field.appendChild(label);
      field.appendChild(sel);
      bar.appendChild(field);
      selects[f.key] = sel;
    });
    var checks = {};
    (config.toggles || []).forEach(function (t) {
      var lab = el("label", "check");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.addEventListener("change", function () { toggles[t.key] = box.checked; render(); });
      lab.appendChild(box);
      lab.appendChild(document.createTextNode(t.label));
      bar.appendChild(lab);
      checks[t.key] = box;
    });
    var reset = el("button", null, "Reset");
    reset.type = "button";
    reset.addEventListener("click", function () {
      for (var k in filters) filters[k] = "";
      for (var c in toggles) { toggles[c] = false; checks[c].checked = false; }
      sort = { key: defaultSort.key, dir: defaultSort.dir };
      render();
    });
    bar.appendChild(reset);
    var chip = el("div", "count-chip");
    var chipN = el("strong");
    var chipRest = el("span");
    chip.appendChild(chipN);
    chip.appendChild(document.createTextNode(" "));
    chip.appendChild(chipRest);
    bar.appendChild(chip);
    mount.appendChild(bar);

    var summaryNode = el("div", "summary");
    mount.appendChild(summaryNode);

    var card = el("div", "card table-card");
    var scroll = el("div", "table-scroll");
    var table = document.createElement("table");
    if (config.caption) table.appendChild(el("caption", null, config.caption));
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    thead.appendChild(headRow);
    table.appendChild(thead);
    var body = document.createElement("tbody");
    table.appendChild(body);
    scroll.appendChild(table);
    var empty = el("div", "empty", "No " + (config.noun || "rows") + " match these filters.");
    empty.hidden = true;
    scroll.appendChild(empty);
    card.appendChild(scroll);
    mount.appendChild(card);

    function matches(rec, skip) {
      for (var k in filters) {
        if (k !== skip && filters[k] && String(rec[k]) !== filters[k]) return false;
      }
      var ts = config.toggles || [];
      for (var i = 0; i < ts.length; i++) {
        if (toggles[ts[i].key] && !ts[i].test(rec)) return false;
      }
      return true;
    }
    function shown() { return config.records.filter(function (r) { return matches(r, null); }); }
    function rebuildSelect(f) {
      var sel = selects[f.key];
      var counts = {};
      config.records.forEach(function (r) {
        if (matches(r, f.key) && r[f.key] !== undefined && r[f.key] !== null) {
          counts[r[f.key]] = (counts[r[f.key]] || 0) + 1;
        }
      });
      var values = Object.keys(counts).sort(natCompare);
      if (filters[f.key] && values.indexOf(filters[f.key]) < 0) values.push(filters[f.key]);
      sel.textContent = "";
      var total = values.reduce(function (s, v) { return s + (counts[v] || 0); }, 0);
      var all = el("option", null, "All (" + total + ")");
      all.value = "";
      sel.appendChild(all);
      values.forEach(function (v) {
        var o = el("option", null, (f.format ? f.format(v) : v) + " (" + (counts[v] || 0) + ")");
        o.value = v;
        sel.appendChild(o);
      });
      sel.value = filters[f.key];
    }
    function colByKey(key) {
      for (var i = 0; i < columns.length; i++) if (columns[i].key === key) return columns[i];
      return null;
    }
    function sortRecords(recs) {
      var col = colByKey(sort.key);
      var dir = sort.dir;
      var value = col ? (col.sortValue || col.value) : null;
      return recs.slice().sort(function (a, b) {
        var d = 0;
        if (value) {
          var va = value(a), vb = value(b);
          var na = va === null || va === undefined, nb = vb === null || vb === undefined;
          if (na || nb) {
            d = na && nb ? 0 : (na ? 1 : -1);   /* empty last either way */
            if (d !== 0) return d;
          } else if (typeof va === "number" && typeof vb === "number") {
            d = va - vb;
          } else {
            d = natCompare(va, vb);
          }
        }
        if (d !== 0) return d * dir;
        return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
      });
    }
    function renderHead() {
      headRow.textContent = "";
      columns.forEach(function (col) {
        var th = el("th");
        th.scope = "col";
        if (col.sortable) {
          var btn = el("button", null, col.label);
          btn.type = "button";
          btn.appendChild(el("span", "arrow", sort.key === col.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""));
          th.setAttribute("aria-sort", sort.key === col.key ? (sort.dir === 1 ? "ascending" : "descending") : "none");
          btn.addEventListener("click", function () {
            if (sort.key === col.key) sort.dir = -sort.dir;
            else { sort.key = col.key; sort.dir = col.firstDir || 1; }
            render();
          });
          th.appendChild(btn);
        } else {
          th.appendChild(document.createTextNode(col.label));
        }
        headRow.appendChild(th);
      });
    }
    function closeDetail() {
      if (!open) return;
      if (window.Plotly) {
        Array.prototype.forEach.call(open.detailTr.querySelectorAll(".plot-div"), function (div) {
          try { window.Plotly.purge(div); } catch (e) { /* not a plot */ }
        });
      }
      open.detailTr.remove();
      open.tr.classList.remove("expanded");
      var btn = open.tr.querySelector(".expander");
      if (btn) { btn.setAttribute("aria-expanded", "false"); btn.textContent = "▸"; }
      open = null;
    }
    function toggleDetail(rec, tr) {
      var wasOpen = open && open.id === rec.id;
      closeDetail();
      if (wasOpen) return;
      var dtr = el("tr", "detail-row");
      var td = document.createElement("td");
      td.colSpan = columns.length;
      var box = el("div", "detail");
      td.appendChild(box);
      dtr.appendChild(td);
      tr.parentNode.insertBefore(dtr, tr.nextSibling);
      tr.classList.add("expanded");
      var btn = tr.querySelector(".expander");
      if (btn) { btn.setAttribute("aria-expanded", "true"); btn.textContent = "▾"; }
      open = { id: rec.id, tr: tr, detailTr: dtr };
      config.detail(box, rec);
    }
    function renderTable(recs) {
      body.textContent = "";
      var frag = document.createDocumentFragment();
      sortRecords(recs).forEach(function (r) {
        var tr = document.createElement("tr");
        if (config.detail) tr.className = "expandable";
        columns.forEach(function (col) {
          var td = document.createElement("td");
          if (col.cls) td.className = col.cls;
          if (col.render) {
            col.render(td, r);
          } else {
            var v = col.value(r);
            td.textContent = v === null || v === undefined ? "" : v;
          }
          tr.appendChild(td);
        });
        if (config.detail) {
          tr.addEventListener("click", function (ev) {
            if (ev.target.closest("a")) return;
            toggleDetail(r, tr);
          });
        }
        frag.appendChild(tr);
      });
      body.appendChild(frag);
      empty.hidden = recs.length > 0;
    }
    function render() {
      hideTip();
      closeDetail();
      (config.filters || []).forEach(rebuildSelect);
      var recs = shown();
      chipN.textContent = recs.length.toLocaleString("en-US");
      chipRest.textContent = "of " + config.records.length.toLocaleString("en-US") + " " + (config.noun || "rows");
      if (config.summary) {
        summaryNode.textContent = "";
        config.summary(recs, summaryNode);
      }
      renderHead();
      renderTable(recs);
    }
    render();
    return {
      render: render,
      shown: shown,
      filters: filters,
      setFilter: function (key, value) { filters[key] = value; render(); }
    };
  }

  vv.el = el;
  vv.num = num;
  vv.fmtInt = fmtInt;
  vv.fmtPct = fmtPct;
  vv.fmtNum = fmtNum;
  vv.fmtDuration = fmtDuration;
  vv.quantile = quantile;
  vv.natCompare = natCompare;
  vv.tile = tile;
  vv.domTable = domTable;
  vv.attachTip = attachTip;
  vv.hideTip = hideTip;
  vv.decodeNp = decodeNp;
  vv.fetchRecord = fetchRecord;
  vv.loadIndex = loadIndex;
  vv.runLine = runLine;
  vv.ensurePlotly = ensurePlotly;
  vv.isDark = isDark;
  vv.plotlyLayout = plotlyLayout;
  vv.axisDefaults = axisDefaults;
  vv.stepXY = stepXY;
  vv.SERIES_COLORS = SERIES_COLORS;
  vv.explorer = explorer;
  window.vv = vv;
})();
