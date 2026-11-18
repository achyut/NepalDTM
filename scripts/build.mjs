// Build the website into dist/: copy site/ and the paper PDFs, and fill every {{name}} in the HTML
// from the validation files (v2: validation_v1_v2.json, validation_report_v2.json, strip_adjust_v2.json;
// v1: validation_report.json, train_report.json), so the site quotes exactly what the validation step
// measured. Fails if a placeholder is left unfilled.
//
//   node scripts/build.mjs

import {cpSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(ROOT, 'dist');
const json = p => JSON.parse(readFileSync(join(ROOT, p), 'utf8'));

const report = json('validation/validation_report.json');
const train = json('validation/train_report.json');
const groups = report.heldout.groups;
const all = groups['all (in Nepal)'];
// v2: every model on the same held-out ATL03 ground photons (tracks never used to build v2)
const cmp = json('validation/validation_v1_v2.json');
const ph = cmp.photons_holdout_dtm;
const rep2 = json('validation/validation_report_v2.json');
const g2 = rep2.heldout.groups;
const adj = json('validation/strip_adjust_v2.json');

const int = n => Math.round(n).toLocaleString('en-US');
const m = (v, d = 2) => v.toFixed(d);
const signed = (v, d = 2) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(d)}`;
const pct = v => Math.round(v * 100);

const dgps = report.dgps['dem/margins'];
const values = {
    rmse: m(all.ours_spatial.rmse),
    rmse_cop: m(all.copernicus.rmse),
    rmse_gain: Math.round((1 - all.ours_spatial.rmse / all.copernicus.rmse) * 100),
    bias: signed(all.ours_spatial.bias),
    bias_cop: signed(all.copernicus.bias),
    n_heldout: int(all.ours_spatial.n),
    n_common: int(report.heldout.n_common),
    n_icesat2: int(train.points.icesat2),
    n_gedi: int(train.points.gedi),
    n_points_m: ((train.points.icesat2 + train.points.gedi) / 1e6).toFixed(1),
    tiles: int(train.points.tiles),
    km2: int(report.coverage.km2_in_tiles),
    share_hma: pct(report.coverage.share_hma_8m),
    share_fill: pct(report.coverage.share_cop_fill),
    err_p50: report.coverage.expected_abs_error_m.p50,
    err_p90: report.coverage.expected_abs_error_m.p90,
    rmse_hma: m(groups['on HMA 8 m'].ours_spatial.rmse),
    rmse_fill: m(groups['Copernicus-filled'].ours_spatial.rmse),
    rmse_steep: m(groups['slope >=40'].ours_spatial.rmse),
    dgps: m(dgps.rmse),
    n_dgps: int(dgps.n),
    // v2 (headline)
    v2_rmse: m(ph.all.v2.rmse),
    v2_nmad: m(ph.all.v2.nmad),
    v1_rmse: m(ph.all.v1.rmse),
    v2_cop_rmse: m(ph.all.copernicus.rmse),
    v2_fab_rmse: m(ph.all.fabdem.rmse),
    v2_gedtm_rmse: m(ph.all.gedtm30.rmse),
    v2_gain_v1: Math.round((1 - ph.all.v2.rmse / ph.all.v1.rmse) * 100),
    v2_gain_cop: Math.round((1 - ph.all.v2.rmse / ph.all.copernicus.rmse) * 100),
    v2_n_photons_m: (ph.all.v2.n / 1e6).toFixed(1),
    v2_n_strips: int(adj.n_strips),
    v2_n_photon_ties_m: (adj.n_photon_ties / 1e6).toFixed(1),
    v2_n_overlap_ties_m: (adj.n_overlap_ties / 1e6).toFixed(1),
    v2_share_hma: pct(rep2.coverage.share_hma_8m),
    v2_share_fill: pct(rep2.coverage.share_cop_fill),
    v2_km2: int(rep2.coverage.km2_in_tiles),
    v2_err_p50: rep2.coverage.expected_abs_error_m.p50,
    v2_err_p90: rep2.coverage.expected_abs_error_m.p90,
    v2_rmse_hma: m(g2['on HMA 8 m'].ours_spatial.rmse),
    v2_rmse_fill: m(g2['Copernicus-filled'].ours_spatial.rmse),
    v2_n_atl08: int(g2.all.ours_spatial.n),
    v2_rmse_steep: m(ph['slope >=40'].v2.rmse),
    v1_rmse_steep: m(ph['slope >=40'].v1.rmse),
    v2_lt_dem: m(cmp.langtang_2m.all['v2/dem'].rmse),
    v1_lt_dem: m(cmp.langtang_2m.all['v1/dem'].rmse),
    lt_cop: m(cmp.langtang_2m.all.copernicus.rmse),
    v2_dgps: m(rep2.dgps['dem/margins'].rmse),
};

// Bar chart: NepalDTM against Copernicus by zone, land cover and base data.
const BARS = [
    ['Physiographic zone', [
        ['zone: terai <300 m', 'Terai', 'below 300 m'],
        ['zone: siwalik 300-1000 m', 'Siwalik', '300 – 1,000 m'],
        ['zone: hills 1-2.5 km', 'Hills', '1 – 2.5 km'],
        ['zone: mountains 2.5-4 km', 'Mountains', '2.5 – 4 km'],
        ['zone: high himal >4 km', 'High Himal', 'above 4 km'],
    ]],
    ['Land cover', [['forest', 'Forest', ''], ['non-forest', 'Open ground', '']]],
    ['Base data', [['on HMA 8 m', 'HMA 8 m stereo', ''], ['Copernicus-filled', 'Copernicus fill', '']]],
];
const scale = Math.max(...BARS.flatMap(([, rows]) => rows.map(([k]) => groups[k].copernicus.rmse)));
const bar = (cls, v) => `<div class="bar ${cls}" style="width:${(v / scale * 100).toFixed(1)}%"><span>${m(v, 1)} m</span></div>`;
values.accuracy_bars = '<div class="bars">' + BARS.map(([title, rows]) =>
    `<p class="group-label">${title}</p>` + rows.map(([k, label, sub]) => {
        const g = groups[k];
        return `<div class="bar-row"><div class="bar-label">${label}<small>${sub ? `${sub} · ` : ''}${int(g.ours_spatial.n)} pts</small></div>` +
            `<div class="bar-pair">${bar('ours', g.ours_spatial.rmse)}${bar('cop', g.copernicus.rmse)}</div></div>`;
    }).join('')).join('') + '</div>';

// Full table: every group, every model.
const MODELS = [['ours_spatial', 'NepalDTM'], ['copernicus', 'Copernicus'], ['fabdem', 'FABDEM'], ['gedtm30', 'GEDTM30']];
const label = k => k === 'all (in Nepal)' ? 'All of Nepal' : k.replace(/^(zone|province): /, '').replace(/^./, c => c.toUpperCase());
values.accuracy_table = '<div class="table-scroll"><table><thead><tr><th>Group</th><th>Points</th>' +
    MODELS.map(([, name]) => `<th>${name}</th>`).join('') + '</tr></thead><tbody>' +
    Object.entries(groups).map(([k, g]) => {
        const best = Math.min(...MODELS.map(([id]) => g[id].rmse));
        return `<tr><td>${label(k)}</td><td>${int(g.ours_spatial.n)}</td>` +
            MODELS.map(([id]) => `<td${g[id].rmse === best ? ' class="best"' : ''}>${m(g[id].rmse)}</td>`).join('') + '</tr>';
    }).join('') + '</tbody></table></div>';

// v2 bar chart: v2, v1 and Copernicus on the same held-out photons, by zone, land cover and slope.
const BARS2 = [
    ['Physiographic zone', [
        ['terai <300 m', 'Terai', 'below 300 m'],
        ['siwalik 300-1000 m', 'Siwalik', '300 – 1,000 m'],
        ['hills 1-2.5 km', 'Hills', '1 – 2.5 km'],
        ['mountains 2.5-4 km', 'Mountains', '2.5 – 4 km'],
        ['high himal >4 km', 'High Himal', 'above 4 km'],
    ]],
    ['Land cover', [['tree', 'Tree cover', ''], ['non-tree', 'No tree cover', ''], ['bare/snow', 'Bare ground and snow', '']]],
    ['Slope', [['slope <20', 'Below 20°', ''], ['slope 20-40', '20 – 40°', ''], ['slope >=40', '40° and steeper', '']]],
];
const scale2 = Math.max(...BARS2.flatMap(([, rows]) => rows.map(([k]) => Math.max(ph[k].copernicus.rmse, ph[k].v1.rmse))));
const bar2 = (cls, v) => `<div class="bar ${cls}" style="width:${(v / scale2 * 100).toFixed(1)}%"><span>${m(v, 1)} m</span></div>`;
values.accuracy_bars_v2 = '<div class="bars">' + BARS2.map(([title, rows]) =>
    `<p class="group-label">${title}</p>` + rows.map(([k, label, sub]) => {
        const g = ph[k];
        return `<div class="bar-row"><div class="bar-label">${label}<small>${sub ? `${sub} · ` : ''}${int(g.v2.n)} photons</small></div>` +
            `<div class="bar-pair">${bar2('ours', g.v2.rmse)}${bar2('v1', g.v1.rmse)}${bar2('cop', g.copernicus.rmse)}</div></div>`;
    }).join('')).join('') + '</div>';

// v2 full table: every group, every model, same photons.
const MODELS2 = [['v2', 'NepalDTM v2'], ['v1', 'NepalDTM v1'], ['copernicus', 'Copernicus'], ['fabdem', 'FABDEM'], ['gedtm30', 'GEDTM30']];
const label2 = k => k === 'all' ? 'All of Nepal' : k.replace(/^./, c => c.toUpperCase());
values.accuracy_table_v2 = '<div class="table-scroll"><table><thead><tr><th>Group</th><th>Photons</th>' +
    MODELS2.map(([, name]) => `<th>${name}</th>`).join('') + '</tr></thead><tbody>' +
    Object.entries(ph).map(([k, g]) => {
        const best = Math.min(...MODELS2.map(([id]) => g[id].rmse));
        return `<tr><td>${label2(k)}</td><td>${int(g.v2.n)}</td>` +
            MODELS2.map(([id]) => `<td${g[id].rmse === best ? ' class="best"' : ''}>${m(g[id].rmse)}</td>`).join('') + '</tr>';
    }).join('') + '</tbody></table></div>';

rmSync(DIST, {recursive: true, force: true});
cpSync(join(ROOT, 'site'), DIST, {recursive: true});
mkdirSync(join(DIST, 'paper'), {recursive: true});
for (const f of ['paper.pdf', 'paper_v2.pdf', 'whitepaper.pdf']) cpSync(join(ROOT, 'paper', f), join(DIST, 'paper', f));

for (const f of readdirSync(DIST).filter(f => f.endsWith('.html'))) {
    const p = join(DIST, f);
    const html = readFileSync(p, 'utf8').replace(/\{\{(\w+)\}\}/g, (_, k) => {
        if (!(k in values)) throw new Error(`${f}: no value for {{${k}}}`);
        return String(values[k]);
    });
    writeFileSync(p, html);
}
console.log(`built ${DIST}`);
