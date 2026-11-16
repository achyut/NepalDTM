// Build the website into dist/: copy site/ and the paper PDFs, and fill every {{name}} in the HTML
// from validation/validation_report.json and validation/train_report.json, so the site quotes
// exactly what the validation step measured. Fails if a placeholder is left unfilled.
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
