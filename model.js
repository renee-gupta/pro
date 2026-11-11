// Core maths: data prep, weighted logistic regression (IRLS), metrics. Works in browser and Node.
const KEEP = { 'Fully Paid': 0, 'Charged Off': 1, 'Default': 1 }; // 'Current'/'In Grace Period' dropped (unresolved)
const FEATS = { annual_inc: 'Annual income ($)', dti: 'Debt-to-income (%)', fico_range_low: 'FICO score', loan_amnt: 'Loan amount ($)', lti: 'Loan-to-income ratio' };

function parseCSV(t) {
  const [h, ...ls] = t.trim().split(/\r?\n/), c = h.split(',');
  return ls.map(l => { const v = l.split(','); return Object.fromEntries(c.map((k, i) => [k, v[i]])); });
}
function median(a) {
  a = a.filter(x => !isNaN(x)).sort((x, y) => x - y);
  return a.length % 2 ? a[a.length >> 1] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2;
}
function prepare(raw) {
  const kept = raw.filter(r => Object.hasOwn(KEEP, r.loan_status) && +r.annual_inc > 0);
  const D = kept.map(r => ({ y: KEEP[r.loan_status], annual_inc: +r.annual_inc, dti: r.dti === '' ? NaN : +r.dti, fico_range_low: +r.fico_range_low, loan_amnt: +r.loan_amnt }));
  const m = median(D.map(r => r.dti));
  D.forEach(r => { if (isNaN(r.dti)) r.dti = m; r.lti = r.loan_amnt / r.annual_inc; });
  return { D, dropped: raw.length - D.length };
}
function inv(A) { // Gauss-Jordan inverse
  const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => +(i === j))]);
  for (let i = 0; i < n; i++) {
    let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    const d = M[i][i]; for (let j = 0; j < 2 * n; j++) M[i][j] /= d;
    for (let r = 0; r < n; r++) if (r !== i) { const f = M[r][i]; for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[i][j]; }
  }
  return M.map(r => r.slice(n));
}
// X rows include the intercept column. cw = [weight for class 0, weight for class 1]
function fit(X, y, cw) {
  const n = X.length, k = X[0].length; let b = Array(k).fill(0), Hi;
  for (let it = 0; it < 50; it++) {
    const g = Array(k).fill(0), H = Array.from({ length: k }, () => Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      let z = 0; for (let j = 0; j < k; j++) z += X[i][j] * b[j];
      const p = 1 / (1 + Math.exp(-z)), w = cw[y[i]];
      for (let j = 0; j < k; j++) { g[j] += w * (y[i] - p) * X[i][j]; for (let l = 0; l < k; l++) H[j][l] += w * p * (1 - p) * X[i][j] * X[i][l]; }
    }
    Hi = inv(H); let m = 0;
    for (let j = 0; j < k; j++) { let d = 0; for (let l = 0; l < k; l++) d += Hi[j][l] * g[l]; b[j] += d; m = Math.max(m, Math.abs(d)); }
    if (m < 1e-9) break;
  }
  return { b, se: Hi.map((r, i) => Math.sqrt(r[i])) };
}
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return Math.sign(x) * (1 - poly * Math.exp(-x * x));
}
const pValue = z => 2 * (1 - 0.5 * (1 + erf(Math.abs(z) / Math.SQRT2)));
function roc(score, y) { // returns {auc, pts}
  const idx = score.map((_, i) => i).sort((a, b) => score[b] - score[a]);
  const P = y.reduce((s, v) => s + v, 0), N = y.length - P;
  let tp = 0, fp = 0, auc = 0; const pts = [[0, 0]];
  for (const i of idx) { if (y[i]) tp++; else { fp++; auc += tp / P / N; } pts.push([fp / N, tp / P]); }
  return { auc, pts };
}
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// Train on a stratified split with the chosen features. Returns everything the UI needs.
function train(D, feats, { balanced, testFrac, seed = 42 }) {
  const rnd = mulberry(seed), tr = [], te = [];
  for (const c of [0, 1]) {
    const idx = D.map((_, i) => i).filter(i => D[i].y === c);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const nt = Math.round(idx.length * testFrac); te.push(...idx.slice(0, nt)); tr.push(...idx.slice(nt));
  }
  const mu = feats.map(f => tr.reduce((s, i) => s + D[i][f], 0) / tr.length);
  const sd = feats.map((f, j) => Math.sqrt(tr.reduce((s, i) => s + (D[i][f] - mu[j]) ** 2, 0) / tr.length) || 1);
  const row = r => [1, ...feats.map((f, j) => (r[f] - mu[j]) / sd[j])];
  const ytr = tr.map(i => D[i].y), n1 = ytr.reduce((s, v) => s + v, 0), n0 = ytr.length - n1;
  const cw = balanced ? [ytr.length / (2 * n0), ytr.length / (2 * n1)] : [1, 1];
  const { b, se } = fit(tr.map(i => row(D[i])), ytr, cw);
  const predict = r => { const z = row(r).reduce((s, v, j) => s + v * b[j], 0); return 1 / (1 + Math.exp(-z)); };
  const pTr = tr.map(i => predict(D[i])).sort((a, c) => a - c);
  const cuts = [1, 2, 3, 4, 5].map(q => pTr[Math.floor(pTr.length * q / 6)]); // sextile cut-offs -> grades A..F
  const yte = te.map(i => D[i].y), pte = te.map(i => predict(D[i]));
  return { names: ['Intercept', ...feats], b, se, predict, cuts, yte, pte, ntr: tr.length, nte: te.length, ...roc(pte, yte) };
}
if (typeof module !== 'undefined') module.exports = { parseCSV, prepare, train, roc, pValue, FEATS };
