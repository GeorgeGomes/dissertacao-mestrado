// ============================================================================
// Porte fiel, linha a linha, de RelaxedPerceptron.fit (src/relaxed_perceptron.py)
// para JavaScript. Cada `yield` carrega `line` = número da linha do arquivo .py
// que acabou de ser executada. O resultado (w, γ, histórico de γ, avisos) é
// idêntico bit a bit ao Python (verificado em verify.js / gerar.py).
// ============================================================================

// Mersenne Twister MT19937, igual a numpy.random.RandomState(seed) com semente inteira
// (init_genrand) e a permutation() legada (Fisher-Yates com random_interval).
class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624); this.idx = 624;
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const s = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
    }
  }
  twist() {
    const mt = this.mt;
    for (let i = 0; i < 624; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
      mt[i] = (mt[(i + 397) % 624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0)) >>> 0;
    }
    this.idx = 0;
  }
  int32() {
    if (this.idx >= 624) this.twist();
    let y = this.mt[this.idx++];
    y ^= y >>> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >>> 18;
    return y >>> 0;
  }
  interval(max) {
    if (max === 0) return 0;
    let mask = max; mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    let v; while ((v = (this.int32() & mask) >>> 0) > max) {}
    return v;
  }
  permutation(n) {
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = this.interval(i); const t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
}

// d_W(x, c) = Σ_j w_j (x_j - c_j)²      (relaxed_perceptron.py, método d_W)
function dW(x, c, w) { let s = 0; for (let j = 0; j < w.length; j++) { const t = x[j] - c[j]; s += w[j] * (t * t); } return s; }
function norm2(w) { let s = 0; for (let j = 0; j < w.length; j++) s += w[j] * w[j]; return Math.sqrt(s); }
const clip = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));

// Valores usados pelo runner: PERCEPTRON_PARAMS (dissertacao_mestrado.py) + defaults da classe
// + use_best_effort=True (todos os 8 pontos de chamada).
const DEFAULT_PARAMS = { eta: 0.001, C: 1.0, gamma_init: 0.1, delta_gamma: 0.05, max_epochs: 50, tol: 1e-4, max_iterations: 50, use_best_effort: true, random_state: 42 };

function* fitGen(X, y, centroids, params) {
  const p = Object.assign({}, DEFAULT_PARAMS, params);
  if (p.delta_gamma <= 0) throw new Error(`delta_gamma deve ser > 0 (recebido: ${p.delta_gamma})`);
  const m = X.length, d = X[0].length;
  const lambda_ = 1.0 / p.C;                              // L159
  let w = new Array(d).fill(1.0);                         // L162
  let alpha = new Array(m).fill(0.0);                     // L166
  let gamma = p.gamma_init;                               // L167
  let gamma_lo = 0.0, gamma_hi = Infinity;                // L173-174
  let w_star = w.slice(), alpha_star = alpha.slice();     // L177-178
  let w_best_effort = w.slice();                          // L182
  let min_global_violations = Infinity;                   // L183
  let found_feasible_solution = false;                    // L184
  let iteration = 0;                                      // L186
  const gamma_history = [];                               // L188
  const rng = new MT19937(p.random_state);                // L191
  const S = () => ({ w: w.slice(), alpha, gamma, gamma_lo, gamma_hi, w_star: w_star.slice(), w_best_effort: w_best_effort.slice(),
                     min_global_violations, found_feasible_solution, iteration, gamma_history, lambda_ });

  while (iteration < p.max_iterations) {                  // L198
    iteration += 1;                                       // L199
    yield Object.assign({ type: 'iter_start', line: 199 }, S());
    let final_violations = -1;                            // L205
    let epochs_run = 0;
    for (let epoch = 0; epoch < p.max_epochs; epoch++) {  // L210
      let violations = 0;                                 // L211
      const indices = rng.permutation(m);                 // L214
      epochs_run = epoch + 1;
      for (let t = 0; t < indices.length; t++) {          // L216
        const i = indices[t];
        const xi = X[i], yi = y[i] | 0;                   // L217
        const distances = centroids.map(c => dW(xi, c, w)); // L220
        const l = yi;                                     // L225
        let k = 0, best = Infinity;                       // L226-228  k = argmin_{j≠l}
        for (let j = 0; j < centroids.length; j++) { if (j === l) continue; if (distances[j] < best) { best = distances[j]; k = j; } }
        let norm_w = norm2(w); if (norm_w < 1e-9) norm_w = 1e-9;  // L231-233
        const margin = dW(xi, centroids[k], w) - dW(xi, centroids[l], w) + lambda_ * alpha[i]; // L242-246
        const violated = margin < gamma * norm_w;         // L248
        let g = null, factor = null;
        if (violated) {
          violations += 1;                                // L250
          g = xi.map((v, j) => { const a = v - centroids[l][j], b = v - centroids[k][j]; return a * a - b * b; }); // L258
          factor = 1 - p.eta * gamma / norm_w;            // L269
          factor = clip(factor, 0.0, 2.0);                // L270
          w = w.map((v, j) => v * factor - p.eta * g[j]); // L272
          w = w.map(v => Math.max(0, v));                 // L275
          alpha = alpha.map(a => a * factor);             // L280
          alpha[i] = alpha[i] + p.eta;                    // L281
          alpha = alpha.map(a => clip(a, 0, 1e6));        // L282
        }
        yield Object.assign({ type: 'point', line: violated ? 272 : 248, i, k, l, epoch, t, d_k: distances[k], d_l: distances[l],
          margin, threshold: gamma * norm_w, norm_w, violated, violations, g, factor, lambda_alpha_i: lambda_ * alpha[i] }, S());
      }
      final_violations = violations;                      // L284
      if (violations < min_global_violations) {           // L288
        min_global_violations = violations; w_best_effort = w.slice(); // L289-290
      }
      yield Object.assign({ type: 'epoch_end', line: violations === 0 ? 296 : 284, epoch, violations, converged: violations === 0 }, S());
      if (violations === 0) break;                        // L293-296
    }
    const viable_now = (final_violations === 0);          // L299
    gamma_history.push({ iter: iteration, gamma, gamma_lo, gamma_hi: isFinite(gamma_hi) ? gamma_hi : null,
                         violations: final_violations, viable: viable_now, epochs: epochs_run }); // L300-307
    let branch, line;
    if (viable_now) {                                     // L310
      found_feasible_solution = true;                     // L313
      gamma_lo = gamma;                                   // L314
      w_star = w.slice(); alpha_star = alpha.slice();     // L315-316
      if (!isFinite(gamma_hi)) { gamma = gamma + p.delta_gamma; branch = 'viavel_incremento'; line = 323; }           // L322-323
      else { gamma = Math.min(gamma + p.delta_gamma, (gamma_lo + gamma_hi) / 2); branch = 'viavel_min'; line = 325; } // L325
    } else {
      gamma_hi = gamma;                                   // L328
      if (isFinite(gamma_hi)) {                           // L330 (sempre verdadeiro: gamma é finito)
        gamma = (gamma_lo + gamma_hi) / 2;                // L333
        w = w_star.slice(); alpha = alpha_star.slice();   // L334-335
        branch = 'inviavel_bissecao'; line = 333;
      } else {                                            // L338 (inalcançável)
        branch = 'inviavel_break';
        yield Object.assign({ type: 'iter_end', line: 338, branch, viable_now, final_violations, stop: true }, S());
        break;
      }
    }
    const stop = Math.abs(gamma_hi - gamma_lo) <= p.tol;  // L341
    yield Object.assign({ type: 'iter_end', line: stop ? 342 : line, branch, viable_now, final_violations, stop }, S());
    if (stop) break;                                      // L342
  }
  // ---- retorno (L348-388) ----
  const bracket_open = Math.abs(gamma_hi - gamma_lo) > p.tol;
  const warnings = [];
  const hiStr = isFinite(gamma_hi) ? gamma_hi.toFixed(6) : '∞';
  if (found_feasible_solution && bracket_open)            // L348
    warnings.push(`Perceptron: busca binária em γ parou por max_iterations=${p.max_iterations} sem convergir (γ_lo=${gamma_lo.toFixed(6)}, γ_hi=${hiStr}, tol=${p.tol}). Retornando melhor γ viável encontrado.`);
  let result, line;
  if (found_feasible_solution) {                          // L361
    result = { w: w_star.slice(), gamma: gamma_lo, caminho: 'viavel' }; line = 365;   // L365
  } else if (p.use_best_effort) {                         // L368
    warnings.push(`Perceptron: separação perfeita não encontrada. Retornando melhor esforço (violações: ${min_global_violations}). Métrica pode ser imprecisa.`);
    result = { w: w_best_effort.slice(), gamma: 0.0, caminho: 'melhor_esforco' }; line = 378; // L378
  } else {
    warnings.push(`Perceptron: separação perfeita não encontrada. Retornando pesos iniciais (equivalente à distância Euclidiana). Verifique a separabilidade dos dados do LLM.`);
    result = { w: w_star.slice(), gamma: 0.0, caminho: 'pesos_iniciais' }; line = 388;  // L388
  }
  const termination = (iteration >= p.max_iterations && bracket_open) ? 'max_iterations' : 'tolerancia';
  yield Object.assign({ type: 'done', line, result, warnings, termination, bracket_open }, S());
  return result;
}

function fitFull(X, y, centroids, params) {
  const g = fitGen(X, y, centroids, params); let r;
  for (;;) { const n = g.next(); if (n.done) return r; if (n.value.type === 'done') r = n.value; }
}
// compute_centroids (src/metrics.py): média por classe, classes em ordem crescente
function computeCentroids(X, y) {
  const cls = [...new Set(y)].sort((a, b) => a - b);
  return cls.map(c => { const rows = X.filter((_, i) => y[i] === c); const d = X[0].length;
    return Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length); });
}

// ============================================================================
// Versão 2: Algoritmo 1 do artigo, versão de 02/08/2026
// (imagem no .docx; transcrito na página). Pontos em ordem i = 1..N, T épocas
// fixas, correção só quando ŷ ≠ y, sem α. O passo 7 ("f ← busca binária em
// [0,1] satisfazendo a relaxação de margem γ") não é definido operacionalmente
// no artigo. Padrão (f_mode = 'um'): f = 1, isto é, o passo inteiro
// w ← max(0, w − η g), a leitura que não inventa uma condição ausente do texto.
// Opção f_mode = 'bissecao': f é o MENOR valor em [0,1] (bisseção) para o qual,
// após a correção w' = max(0, w − η f g), vale d_W'(x, μ_ŷ) − d_W'(x, μ_y) ≥ γ‖w'‖
// (se nem f = 1 satisfaz, usa-se f = 1). Essa condição é invariante à escala de
// w' e aceita w' ≈ 0, por isso w colapsa para ~0 em 4 dos 6 conjuntos reais;
// excluir só w' = 0 não muda nenhum resultado (testado em 01/10/2026).
// ============================================================================
const DEFAULT_PARAMS_ARTIGO = { eta: 0.001, gamma: 0.1, T: 50, f_mode: 'um', bissecao_passos: 20 };
function* fitGenArtigo(X, y, centroids, params) {
  const p = Object.assign({}, DEFAULT_PARAMS_ARTIGO, params);
  const m = X.length, d = X[0].length;
  let w = new Array(d).fill(1.0);                                   // linha 1
  const S = () => ({ w: w.slice(), gamma: p.gamma, alpha: null });
  let total_epochs = 0;
  for (let t = 1; t <= p.T; t++) {                                  // linha 2
    yield Object.assign({ type: 'iter_start', line: 2, t, epoch: t - 1 }, S());
    let errors = 0;
    for (let i = 0; i < m; i++) {                                   // linha 3
      const xi = X[i], yi = y[i] | 0;
      const dists = centroids.map(c => dW(xi, c, w));
      let yhat = 0; for (let k = 1; k < dists.length; k++) if (dists[k] < dists[yhat]) yhat = k; // linha 4
      const mis = yhat !== yi;                                      // linha 5
      let g = null, f = null, f_viavel = null;
      if (mis) {
        errors++;
        g = xi.map((v, j) => { const a = v - centroids[yi][j], b = v - centroids[yhat][j]; return a * a - b * b; }); // linha 6
        if (p.f_mode === 'um') { f = 1; f_viavel = null; }         // linha 7 (padrão: f = 1)
        else {                                                      // linha 7 (opção: busca binária)
          const ok = ff => { const w2 = w.map((v, j) => Math.max(0, v - p.eta * ff * g[j])); return dW(xi, centroids[yhat], w2) - dW(xi, centroids[yi], w2) >= p.gamma * norm2(w2); };
          if (!ok(1)) { f = 1; f_viavel = false; }
          else { let lo = 0, hi = 1; for (let s = 0; s < p.bissecao_passos; s++) { const mid = (lo + hi) / 2; if (ok(mid)) hi = mid; else lo = mid; } f = hi; f_viavel = true; }
        }
        w = w.map((v, j) => Math.max(0, v - p.eta * f * g[j]));     // linha 8
      }
      yield Object.assign({ type: 'point', line: mis ? 8 : 5, i, t: i, epoch: t - 1, k: yhat, l: yi, d_k: dists[yhat], d_l: dists[yi], violated: mis, violations: errors, g, f, f_viavel }, S());
    }
    total_epochs = t;
    yield Object.assign({ type: 'epoch_end', line: 10, epoch: t - 1, violations: errors, converged: false }, S()); // linha 10
  }
  yield Object.assign({ type: 'done', line: 12, result: { w: w.slice(), gamma: p.gamma, caminho: 'T_epocas' }, termination: 'T_epocas', warnings: [], total_epochs }, S()); // linha 12
  return { w };
}

// ============================================================================
// Versão 3: Algoritmo 1, versão revisada (documento de
// 28/08/2026), transcrito literalmente na página. Diferenças em relação ao texto,
// necessárias para executar: ‖w‖₂ é protegida contra zero (1e-12) e o termo de
// correção escalar η·(d_w(v,c_ŷ) − d_w(v,c_y)) é subtraído de cada componente de w
// (leitura literal; termo = 'vetor' usa o gradiente componente a componente
// (v−c_ŷ)² − (v−c_y)²). O fator (1 − ηγ/‖w‖₂) da linha de α é recalculado com o w
// já atualizado, como a ordem das linhas indica. Não há projeção w ≥ 0 no texto
// (projetar = false); a saída declara w ∈ R²₊, por isso existe a opção.
// ============================================================================
const DEFAULT_PARAMS_PROFESSOR = { eta: 0.001, gamma: 0.1, T: 50, termo: 'escalar', projetar: false };
function* fitGenProfessor(X, y, centroids, params) {
  const p = Object.assign({}, DEFAULT_PARAMS_PROFESSOR, params);
  const m = X.length, d = X[0].length;
  let w = new Array(d).fill(1.0), alpha = new Array(m).fill(0.0), t = 1, stop = false; // linha 6
  const S = () => ({ w: w.slice(), gamma: p.gamma, alpha, t, stop });
  let total_epochs = 0;
  while (t <= p.T && !stop) {                                       // linha 7
    stop = true;                                                    // linha 8
    yield Object.assign({ type: 'iter_start', line: 8, epoch: t - 1 }, S());
    let errors = 0;
    for (let i = 0; i < m; i++) {                                   // linha 9
      const xi = X[i], yi = y[i] | 0;
      const dists = centroids.map(c => dW(xi, c, w));
      let yhat = 0; for (let k = 1; k < dists.length; k++) if (dists[k] < dists[yhat]) yhat = k; // linha 10
      const mis = yi !== yhat;                                      // linha 11
      let factor = null, termo = null, factor2 = null;
      if (mis) {
        stop = false; errors++;                                     // linha 12
        let nw = norm2(w); if (nw < 1e-12) nw = 1e-12;
        factor = 1 - p.eta * p.gamma / nw;
        if (p.termo === 'escalar') { termo = dW(xi, centroids[yhat], w) - dW(xi, centroids[yi], w); w = w.map(v => v * factor - p.eta * termo); } // linha 13
        else { termo = xi.map((v, j) => { const a = v - centroids[yhat][j], b = v - centroids[yi][j]; return a * a - b * b; }); w = w.map((v, j) => v * factor - p.eta * termo[j]); }
        if (p.projetar) w = w.map(v => Math.max(0, v));
        let nw2 = norm2(w); if (nw2 < 1e-12) nw2 = 1e-12;
        factor2 = 1 - p.eta * p.gamma / nw2;
        alpha = alpha.map(a => a * factor2);                        // linha 14
        alpha[i] = alpha[i] + p.eta;                                // linha 15
      }
      yield Object.assign({ type: 'point', line: mis ? 13 : 11, i, t: i, epoch: t - 1, k: yhat, l: yi, d_k: dists[yhat], d_l: dists[yi], violated: mis, violations: errors, factor, factor2, termo }, S());
    }
    total_epochs = t;
    yield Object.assign({ type: 'epoch_end', line: 18, epoch: t - 1, violations: errors, converged: stop }, S());
    t = t + 1;                                                      // linha 18
  }
  yield Object.assign({ type: 'done', line: 20, result: { w: w.slice(), gamma: p.gamma, caminho: stop ? 'sem_erro' : 'T_epocas' }, termination: stop ? 'sem_erro' : 'T_epocas', warnings: [], total_epochs }, S()); // linha 20
  return { w };
}

if (typeof module !== 'undefined') module.exports = { MT19937, fitGen, fitFull, computeCentroids, dW, norm2, DEFAULT_PARAMS, fitGenArtigo, fitGenProfessor, DEFAULT_PARAMS_ARTIGO, DEFAULT_PARAMS_PROFESSOR };
