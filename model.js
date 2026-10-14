// Regression math used by index.html when the Flask backend is unreachable.
// Reads globals set by index.html: G, c, Xtr, ytr, Xte, yte, sum.

// ---------- csv ----------
// ponytail: naive CSV split — no commas inside quoted fields; swap in PapaParse if a dataset needs it
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  const clean = s => s.trim().replace(/^"|"$/g, '');
  const header = lines[0].split(',').map(clean);
  const rows = lines.slice(1).map(l => l.split(',').map(clean)).filter(r => r.length === header.length);
  const numeric = header.map((_, j) => rows.every(r => r[j] !== '' && !isNaN(+r[j])));
  return { header, rows, numeric };
}

// ---------- linear algebra & stats ----------
function inv(M) {
  const n = M.length, A = M.map((r, i) => [...r, ...r.map((_, j) => +(i === j))]);
  for (let col = 0; col < n; col++) {
    let m = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[m][col])) m = r;
    if (Math.abs(A[m][col]) < 1e-9 * Math.max(1, Math.abs(M[col][col]))) return null; // singular / collinear
    [A[col], A[m]] = [A[m], A[col]];
    const p = A[col][col];
    for (let k = 0; k < 2 * n; k++) A[col][k] /= p;
    for (let r = 0; r < n; r++) if (r !== col && A[r][col]) {
      const f = A[r][col];
      for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[col][k];
    }
  }
  return A.map(r => r.slice(n));
}
function gram(X, y) {
  const p = X[0].length, G = Array.from({ length: p }, () => Array(p).fill(0)), c = Array(p).fill(0);
  X.forEach((r, i) => { for (let a = 0; a < p; a++) { c[a] += r[a] * y[i]; for (let b = a; b < p; b++) G[a][b] += r[a] * r[b]; } });
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
  return { G, c };
}
function solveSub(G, c, S) {
  const iM = inv(S.map(a => S.map(b => G[a][b])));
  return iM && { iM, b: iM.map(row => sum(row.map((v, j) => v * c[S[j]]))) };
}
function lgamma(x) {
  const k = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5, s = 1.000000000190015;
  t -= (x + 0.5) * Math.log(t);
  for (const v of k) s += v / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function betacf(a, b, x) {
  const T = 1e-300, fix = v => Math.abs(v) < T ? T : v;
  let c = 1, d = 1 / fix(1 - (a + b) * x / (a + 1)), h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 / fix(1 + aa * d); c = fix(1 + aa / c); h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 / fix(1 + aa * d); c = fix(1 + aa / c);
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < 3e-14) break;
  }
  return h;
}
function ibeta(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const tPval = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));            // two-sided
const fPval = (F, d1, d2) => ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F));
function tCrit(df) { let lo = 0, hi = 100; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; tPval(m, df) > 0.05 ? lo = m : hi = m; } return lo; }

// ---------- feature selection ----------
function standardize(X, y) { // X without intercept column
  const p = X[0].length, n = X.length;
  const mu = [...Array(p).keys()].map(j => sum(X.map(r => r[j])) / n);
  const sd = mu.map((m, j) => Math.sqrt(sum(X.map(r => (r[j] - m) ** 2)) / n) || 1);
  const my = sum(y) / n, sy = Math.sqrt(sum(y.map(v => (v - my) ** 2)) / n) || 1;
  return { Z: X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), yz: y.map(v => (v - my) / sy) };
}
// Lasso (alpha=1) / ElasticNet (alpha=.5) via coordinate descent; rank = order of entry along the λ path
function pathOrder(Z, yz, alpha) {
  const n = Z.length, p = Z[0].length, b = Array(p).fill(0), r = [...yz];
  const lmax = Math.max(...[...Array(p).keys()].map(j => Math.abs(sum(Z.map((row, i) => row[j] * yz[i]))) / n)) / alpha;
  const enter = Array(p).fill(Infinity), mag = Array(p).fill(0);
  for (let s = 0; s < 100; s++) {
    const lam = lmax * Math.pow(1e-4, s / 99);
    for (let it = 0; it < 100; it++) {
      let maxd = 0;
      for (let j = 0; j < p; j++) {
        const rho = sum(Z.map((row, i) => row[j] * r[i])) / n + b[j];
        const nb = Math.sign(rho) * Math.max(Math.abs(rho) - lam * alpha, 0) / (1 + lam * (1 - alpha));
        const d = nb - b[j];
        if (d) { Z.forEach((row, i) => r[i] -= d * row[j]); b[j] = nb; maxd = Math.max(maxd, Math.abs(d)); }
      }
      if (maxd < 1e-7) break;
    }
    b.forEach((v, j) => { if (v && enter[j] === Infinity) { enter[j] = s; mag[j] = Math.abs(v); } });
  }
  return [...Array(p).keys()].sort((i, j) => enter[i] - enter[j] || mag[j] - mag[i]);
}
function rfeOrder(Z, yz) {
  const { G: Gz, c: cz } = gram(Z, yz);
  Gz.forEach((r, i) => r[i] += 1e-9); // tiny ridge so collinear columns still rank
  let S = [...Array(Z[0].length).keys()]; const out = [];
  while (S.length) {
    const b = solveSub(Gz, cz, S).b, i = b.reduce((m, v, j) => Math.abs(v) < Math.abs(b[m]) ? j : m, 0);
    out.unshift(S.splice(i, 1)[0]);
  }
  return out;
}
const sseOf = F => { const S = [0, ...F.map(j => j + 1)], s = solveSub(G, c, S); return s ? sum(ytr.map(v => v * v)) - sum(s.b.map((v, i) => v * c[S[i]])) : Infinity; };
function forwardOrder(p) {
  const out = [], left = [...Array(p).keys()];
  while (left.length) {
    const i = left.reduce((m, j, ii) => sseOf([...out, j]) < sseOf([...out, left[m]]) ? ii : m, 0);
    out.push(left.splice(i, 1)[0]);
  }
  return out;
}
function exhaustiveSubsets(p) {
  if (p > 15) return null; // ponytail: 2^p blowup; falls back to forward SFS above 15 features
  const best = Array(p + 1).fill(null).map(() => ({ sse: Infinity, F: null }));
  for (let m = 1; m < 1 << p; m++) {
    const F = [...Array(p).keys()].filter(j => m >> j & 1), e = sseOf(F);
    if (e < best[F.length].sse) best[F.length] = { sse: e, F };
  }
  return best.slice(1).map(b => b.F);
}

// ---------- model evaluation ----------
function evaluate(F) {
  const S = [0, ...F.map(j => j + 1)], sol = solveSub(G, c, S);
  if (!sol) return null;
  const { b, iM } = sol, pred = X => X.map(r => sum(S.map((a, i) => r[a] * b[i])));
  const yhTr = pred(Xtr), yhTe = pred(Xte), n = ytr.length, k = F.length, df = n - k - 1;
  const mTr = sum(ytr) / n, sse = sum(ytr.map((v, i) => (v - yhTr[i]) ** 2)), sst = sum(ytr.map(v => (v - mTr) ** 2));
  const sigma2 = sse / df, fstat = ((sst - sse) / k) / sigma2;
  const nt = yte.length, mTe = sum(yte) / nt, res = yte.map((v, i) => v - yhTe[i]);
  const r2 = 1 - sum(res.map(v => v * v)) / sum(yte.map(v => (v - mTe) ** 2));
  const se = b.map((_, i) => Math.sqrt(sigma2 * iM[i][i])), t = b.map((v, i) => v / se[i]), tc = tCrit(df);
  return {
    F, b, se, t, p: t.map(v => tPval(Math.abs(v), df)), lo: b.map((v, i) => v - tc * se[i]), hi: b.map((v, i) => v + tc * se[i]),
    vif: vif(F), yhTe, df, fstat, fp: fPval(fstat, k, df),
    rmse: Math.sqrt(sum(res.map(v => v * v)) / nt), r2, mae: sum(res.map(Math.abs)) / nt,
    adj: nt - k - 1 > 0 ? 1 - (1 - r2) * (nt - 1) / (nt - k - 1) : NaN,
  };
}
function vif(F) {
  if (F.length === 1) return [1];
  const { Z } = standardize(Xtr.map(r => F.map(j => r[j + 1])), ytr), n = Z.length;
  const R = F.map((_, a) => F.map((_, b) => sum(Z.map(r => r[a] * r[b])) / n)), iR = inv(R);
  return iR ? iR.map((r, i) => r[i]) : F.map(() => Infinity);
}
