
// ===== Utilities =====
async function loadMarkdownTable(url){
  const resp = await fetch(url, {cache: "no-store"});
  if(!resp.ok) throw new Error("Failed to load " + url + " (" + resp.status + ")");
  const text = await resp.text();
  return parseMarkdownTable(text);
}

function parseMarkdownTable(md){
  const lines = md.split(/\r?\n/);
  // capture first contiguous table block (lines starting with '|')
  let start = -1, end = -1;
  for (let i=0; i<lines.length; i++){
    if(lines[i].trim().startsWith("|")){ start = i; break; }
  }
  if(start === -1) return {headers:[], rows:[]};
  end = start;
  while(end < lines.length && lines[end].trim().startsWith("|")) end++;
  const tableLines = lines.slice(start, end).map(l => l.trim()).filter(Boolean);
  if(tableLines.length < 3) return {headers:[], rows:[]};
  const headers = tableLines[0].slice(1, -1).split("|").map(s => s.trim());
  const rowLines = tableLines.slice(2);
  const rows = rowLines.map(line => {
    const cells = line.slice(1, -1).split("|").map(s => s.trim());
    const obj = {};
    headers.forEach((h, idx) => { obj[h] = cells[idx]; });
    return obj;
  });
  return {headers, rows};
}

function unique(arr){ return [...new Set(arr)]; }
function byNumber(a,b){ return a-b; }
function fmt(n){
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, {maximumFractionDigits: 2});
}

// Simplify model name heuristics; unify 'Meta-Llama' -> 'Llama-'
function simplifyModelName(s){
  if(!s) return s;
  let n = String(s);
  const segs = n.split('--'); n = segs[segs.length - 1]; // keep last segment
  n = n.replace(/_/g, '-').replace(/\s+/g, '-');
  // unify Meta-Llama prefix
  n = n.replace(/^meta[-\s]?llama[-\s]?/i, 'Llama-');
  n = n.replace(/^Meta[-\s]?Llama[-\s]?/i, 'Llama-');
  // remove suffix noise
  n = n
    .replace(/-?Instruct(-v?\d+)?$/i, '')
    .replace(/-?Chat$/i, '')
    .replace(/-?HF$/i, '')
    .replace(/-?GPTQ/i, '')
    .replace(/-?AWQ/i, '')
    .replace(/-?Q\d{2}$/i, '');
  n = n.replace(/--+/g, '-').replace(/^-|-$/g, '');
  return n;
}

// Build groups per Target: [ AR baseline row, ... PEARL rows ]
function buildGroups(rows, batchSize, dataset){
  const filtered = rows.filter(r => String(r["Batch Size"]).trim() === String(batchSize).trim()
                                 && String(r["Benchmark"]).trim() === String(dataset).trim());
  const groups = new Map();
  for(const r of filtered){
    const tgt = r["Target Model"];
    if(!groups.has(tgt)) groups.set(tgt, {target: tgt, pearls: [], ar: null});
    if(r["Mode"] === "AR") groups.get(tgt).ar = r;
    else if(r["Mode"] === "PEARL") groups.get(tgt).pearls.push(r);
  }
  // Order: sort targets by best speedup desc (if AR and PEARL present)
  const order = Array.from(groups.values()).sort((A,B) => {
    const a_ar = A.ar ? Number(A.ar["Throughput (tok/s)"]) : 0;
    const b_ar = B.ar ? Number(B.ar["Throughput (tok/s)"]) : 0;
    const a_best = Math.max(...A.pearls.map(p => Number(p["Throughput (tok/s)"]) / (a_ar||1) || 0), 0);
    const b_best = Math.max(...B.pearls.map(p => Number(p["Throughput (tok/s)"]) / (b_ar||1) || 0), 0);
    return (b_best - a_best);
  });
  return order;
}

function colorFor(str){
  let h = 0;
  for(let i=0;i<str.length;i++){
    h = (h * 31 + str.charCodeAt(i)) >>> 0;
  }
  const hue = h % 360;
  const sat = 55 + ((h >> 8) % 20);   // 55% - 74%
  const light = 68 + ((h >> 16) % 10); // 68% - 77%
  return `hsl(${hue}deg, ${sat}%, ${light}%)`;
}

const state = { data:null, batch:null, dataset:null, chart:null, flatRows: [], chartVisible:false };

function populateSelects(rows){
  const batches = unique(rows.map(r => r["Batch Size"])).map(x => Number(x)).sort(byNumber);
  const datasets = unique(rows.map(r => r["Benchmark"]));
  const bsel = document.getElementById("batch-select");
  const dsel = document.getElementById("dataset-select");
  bsel.innerHTML = batches.map(b => `<option value="${b}">${b}</option>`).join("");
  dsel.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join("");
  state.batch = batches[0]; state.dataset = datasets[0];
  bsel.value = String(state.batch); dsel.value = state.dataset;
  bsel.addEventListener("change", () => { state.batch = Number(bsel.value); render(); });
  dsel.addEventListener("change", () => { state.dataset = dsel.value; render(); });

  const enhanceSelectHitArea = (sel) => {
    const wrapper = sel.closest(".select");
    if(!wrapper) return;
    wrapper.addEventListener("click", (evt) => {
      if(evt.target === sel) return;
      sel.focus({preventScroll:true});
      if(typeof sel.showPicker === "function"){
        try{ sel.showPicker(); }catch(_) { /* noop for unsupported browsers */ }
      }
    });
  };

  enhanceSelectHitArea(bsel);
  enhanceSelectHitArea(dsel);
}

function setChartVisibility(show){
  state.chartVisible = !!show;
  const card = document.getElementById("chart-card");
  const tableCard = document.getElementById("table-card");
  const btn = document.getElementById("visualize-toggle");
  if(card){
    card.classList.toggle("is-hidden", !state.chartVisible);
    card.setAttribute("aria-hidden", state.chartVisible ? "false" : "true");
  }
  if(tableCard){
    tableCard.classList.toggle("is-hidden", state.chartVisible);
    tableCard.setAttribute("aria-hidden", state.chartVisible ? "true" : "false");
  }
  if(btn){
    btn.classList.toggle("is-active", state.chartVisible);
    btn.setAttribute("aria-expanded", state.chartVisible ? "true" : "false");
    btn.textContent = state.chartVisible ? "Hide Visualize" : "Visualize";
  }
  if(state.chartVisible){
    renderChart();
  }else if(state.chart){
    state.chart.destroy();
    state.chart = null;
  }
}

function initVisualToggle(){
  const btn = document.getElementById("visualize-toggle");
  if(!btn) return;
  btn.addEventListener("click", () => {
    setChartVisibility(!state.chartVisible);
  });
}

function renderTable(groups){
  const tbody = document.querySelector("#bench-table tbody");
  const rowsHTML = [];
  state.flatRows = []; // keep a flat list in rendering order for chart

  for(const g of groups){
    const tSimple = simplifyModelName(g.target);
    const grpColor = colorFor(tSimple);
    const ar_tp = g.ar ? Number(g.ar["Throughput (tok/s)"]) : NaN;
    const ar_mat = g.ar ? Number(g.ar["MAT"]) : NaN;
    const pearlRows = g.pearls || [];

    const rowspan = 1 + pearlRows.length;
    // First row: AR baseline
    rowsHTML.push(
      `<tr>
        <th scope="row" class="col-target target-cell" rowspan="${rowspan}" style="--grp:${grpColor}">${tSimple}</th>
        <td class="col-draft">
          <span class="mode-chip ar">AR</span>
          <span class="model-name">None</span>
        </td>
        <td class="col-throughput">${fmt(ar_tp)} tok/s</td>
        <td class="col-mat">${fmt(ar_mat)}</td>
        <td class="speedup-col">1x</td>
      </tr>`
    );

    // Subsequent rows: PEARL lines
    for(const p of pearlRows){
      const dSimple = simplifyModelName(p["Draft Model"]);
      const pearl_tp = Number(p["Throughput (tok/s)"]);
      const mat = Number(p["MAT"]);
      const s = (ar_tp>0 && pearl_tp>0) ? (pearl_tp/ar_tp) : NaN;
      rowsHTML.push(
        `<tr>
          <td class="col-draft">
            <span class="mode-chip pearl">PEARL</span>
            <span class="model-name">${dSimple}</span>
          </td>
          <td class="col-throughput">${fmt(pearl_tp)} tok/s</td>
          <td class="col-mat">${fmt(mat)}</td>
          <td class="speedup-col">${isFinite(s)? fmt(s)+'x':'—'}</td>
        </tr>`
      );
      state.flatRows.push({
        target: tSimple,
        draft: dSimple,
        ar_tps: ar_tp,
        pearl_tps: pearl_tp,
        speedup: s
      });
    }
  }

  tbody.innerHTML = rowsHTML.join("");
}

function setCanvasHeightForRows(n){
  const canvas = document.getElementById("bench-chart");
  const base = 160; // legend + padding
  const per = 40;   // per-row height
  canvas.height = base + Math.max(n, 3) * per;
}

function renderChart(){
  const rows = state.flatRows;
  const canvas = document.getElementById("bench-chart");
  const ctx = canvas.getContext("2d");
  const labels = rows.map(r => [r.target, `→ ${r.draft}`]);
  const dataPearl = rows.map(r => r.pearl_tps || 0);
  const dataAR = rows.map(r => r.ar_tps || 0);
  const speedups = rows.map(r => r.speedup);

  setCanvasHeightForRows(rows.length);

  const styles = getComputedStyle(document.documentElement);
  const textColor = (styles.getPropertyValue('--text') || '#141413').trim() || '#141413';
  const muted = (styles.getPropertyValue('--muted') || '#6B6A64').trim() || '#6B6A64';
  const arBg = (styles.getPropertyValue('--ar-label-bg') || '#def5e8').trim() || '#def5e8';
  const arBorder = (styles.getPropertyValue('--ar-label-border') || '#8fcea7').trim() || '#8fcea7';
  const arText = (styles.getPropertyValue('--ar-label-text') || '#2f6a44').trim() || '#2f6a44';
  const pearlBg = (styles.getPropertyValue('--pearl-label-bg') || '#e1ecff').trim() || '#e1ecff';
  const pearlBorder = (styles.getPropertyValue('--pearl-label-border') || '#9fbaf6').trim() || '#9fbaf6';
  const pearlText = (styles.getPropertyValue('--pearl-label-text') || '#2c4f8b').trim() || '#2c4f8b';

  const withAlpha = (color, alphaHex, fallback) => {
    if(typeof color === "string"){
      const clean = color.trim();
      if(/^#([0-9a-fA-F]{6})$/.test(clean)){
        return `${clean}${alphaHex}`;
      }
      if(/^#([0-9a-fA-F]{8})$/.test(clean)){
        return clean;
      }
    }
    return fallback;
  };

  const arFill = withAlpha(arBg, 'f0', 'rgba(222,245,232,0.85)');
  const pearlFill = withAlpha(pearlBg, 'f0', 'rgba(225,236,255,0.88)');
  const calloutFill = withAlpha(pearlBg, '55', 'rgba(225,236,255,0.35)');
  const calloutStroke = withAlpha(pearlBorder, 'aa', 'rgba(159,186,246,0.65)');

  const datasets = [
    {
      label: 'AR baseline',
      data: dataAR,
      backgroundColor: arFill,
      borderColor: arBorder,
      borderWidth: 1,
      hoverBackgroundColor: withAlpha(arBg, 'ff', 'rgba(222,245,232,1)'),
      hoverBorderColor: arBorder,
      borderRadius: 10,
      barPercentage: 0.5,
      categoryPercentage: 0.6,
      order: 1
    },
    {
      label: 'PEARL throughput',
      data: dataPearl,
      backgroundColor: pearlFill,
      borderColor: pearlBorder,
      borderWidth: 1.5,
      hoverBackgroundColor: withAlpha(pearlBg, 'ff', 'rgba(225,236,255,1)'),
      hoverBorderColor: pearlBorder,
      borderRadius: 10,
      barPercentage: 0.5,
      categoryPercentage: 0.6,
      order: 2
    }
  ];

  const options = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: {
        top: 24,
        right: 56,
        bottom: 24,
        left: 18
      }
    },
    interaction: { mode: 'nearest', axis: 'y', intersect: true },
    scales: {
      x: {
        beginAtZero: true,
        grid: { color: 'rgba(20,20,19,0.08)', drawTicks: false },
        border: { color: 'rgba(20,20,19,0.12)' },
        ticks: {
          font: { size: 11, family: 'Poppins', weight: '500' },
          color: muted
        },
        title: {
          display: true,
          text: 'Throughput (tokens / second)',
          color: muted,
          font: { size: 12, family: 'Poppins', weight: '600' },
          padding: { top: 12 }
        }
      },
      y: {
        grid: { display: false },
        ticks: {
          autoSkip: false,
          font: { size: 12, family: 'Poppins', weight: '600' },
          color: textColor
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'start',
        labels: {
          usePointStyle: true,
          pointStyle: 'rectRounded',
          padding: 18,
          boxWidth: 14,
          font: { size: 12, family: 'Poppins', weight: '600' },
          color: muted
        }
      },
      tooltip: {
        backgroundColor: '#141413',
        titleFont: { size: 13, family: 'Poppins', weight: '600' },
        bodyFont: { size: 12, family: 'Poppins' },
        padding: 12,
        displayColors: true,
        callbacks: {
          title(ctx){
            const lbl = ctx[0].label;
            if(Array.isArray(lbl)) return lbl;
            return [lbl];
          },
          label(ctx){
            return `${ctx.dataset.label}: ${fmt(ctx.parsed.x)} tok/s`;
          },
          afterBody(ctx){
            if(!ctx || !ctx.length) return '';
            const idx = ctx[0].dataIndex;
            const s = speedups[idx];
            return (s && isFinite(s)) ? `Speedup: ${s.toFixed(2)}x` : '';
          }
        }
      }
    },
    animation: { duration: 520, easing: 'easeOutQuart' }
  };

  if(state.chart){
    state.chart.destroy();
  }

  state.chart = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options,
    plugins: [speedupCallouts]
  });
  state.chart.$speedups = speedups;
  state.chart.$pearlDatasetIndex = 1;
  state.chart.$calloutStyles = {
    fill: calloutFill,
    stroke: calloutStroke,
    text: pearlText
  };
}

function roundedRectPath(ctx, x, y, width, height, radius){
  const r = Math.min(radius, width / 2, height / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

const speedupCallouts = {
  id: 'speedupCallouts',
  afterDatasetsDraw(chart){
    const metaIndex = chart.$pearlDatasetIndex;
    if(metaIndex === undefined) return;
    const meta = chart.getDatasetMeta(metaIndex);
    if(!meta || !meta.data) return;
    const speedups = chart.$speedups || [];
    const styles = chart.$calloutStyles || {};
    const fill = styles.fill || 'rgba(217,119,87,0.18)';
    const stroke = styles.stroke || 'rgba(217,119,87,0.45)';
    const textColor = styles.text || '#9a3e1f';
    const ctx = chart.ctx;
    const area = chart.chartArea;

    ctx.save();
    ctx.textBaseline = 'middle';
    ctx.font = '700 12px Poppins, system-ui, sans-serif';
    ctx.textAlign = 'center';

    meta.data.forEach((el, idx) => {
      if(!el || typeof el.x !== 'number' || typeof el.y !== 'number') return;
      const speed = speedups[idx];
      if(!(speed > 0 && isFinite(speed))) return;

      const text = `${speed.toFixed(2)}x`;
      const metrics = ctx.measureText(text);
      const paddingX = 12;
      const height = 24;
      const width = metrics.width + paddingX * 2;
      let x = el.x + 14;
      const y = el.y;

      if(x + width > area.right - 6){
        x = area.right - width - 6;
      }

      ctx.beginPath();
      if(typeof ctx.roundRect === 'function'){
        ctx.roundRect(x, y - height / 2, width, height, 11);
      }else{
        roundedRectPath(ctx, x, y - height / 2, width, height, 11);
      }
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = stroke;
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.fillText(text, x + width / 2, y);
    });

    ctx.restore();
  }
};

function render(){
  const groups = buildGroups(state.data.rows, state.batch, state.dataset);
  renderTable(groups);
  if(state.chartVisible){
    renderChart();
  }else if(state.chart){
    state.chart.destroy();
    state.chart = null;
  }
}

(async function main(){
  try{
    const data = await loadMarkdownTable("./bench_summary.md");
    if(!data || !data.rows || data.rows.length === 0){
      throw new Error("Parsed 0 rows from bench_summary.md");
    }
    state.data = data;
    populateSelects(data.rows);
    initVisualToggle();
    render();
    setChartVisibility(false);
  }catch(err){
    console.error(err);
    const tbody = document.querySelector("#bench-table tbody");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="5" style="color:#b00;font-weight:700">
        Failed to load data: ${String(err.message || err)}
      </td></tr>`;
    }
  }
})();
