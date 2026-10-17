// Terrain maths shared by the page and the tile workers: decoding heights from the Terrarium tiles, and
// computing slope, aspect, roughness and cast shadows tile by tile. No globals, so it runs in a worker.
//
//   slope      Horn's 3 x 3 gradient, degrees
//   aspect     compass direction the slope faces, degrees from north (flat cells, under 2 degrees, are grey)
//   roughness  largest height difference in the 3 x 3 window, metres (GDAL's definition)
//   shadow     ground hidden from the sun by a ridge, as a see-through overlay
//
// Values depend on the pixel size: zoom 14 (8.4 m pixels over Nepal) is the finest.

export const TILE_SIZE = 256;
export const TILE_MAX_ZOOM = 14;
const N = TILE_SIZE;

// Heights from the Terrarium tiles: height = R * 256 + G + B / 256 - 32768. Each tile decodes to 256 x 256
// heights, row by row from the north-west corner, or null where there is no terrain.
// `tileUrl(z, x, y)` gives the address; the worker answers 404 where the archive has no tile.
export function heightSource(tileUrl, keep = 400) {
    const cache = new Map();
    return function tileHeights(z, x, y) {
        const key = `${z}/${x}/${y}`;
        if (!cache.has(key)) {
            if (cache.size > keep) cache.delete(cache.keys().next().value);
            cache.set(key, fetch(tileUrl(z, x, y)).then(async r => {
                if (!r.ok) return null;
                const bmp = await createImageBitmap(await r.blob(), {premultiplyAlpha: 'none', colorSpaceConversion: 'none'});
                const ctx = new OffscreenCanvas(bmp.width, bmp.height).getContext('2d', {willReadFrequently: true});
                ctx.drawImage(bmp, 0, 0);
                const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
                const h = new Float32Array(bmp.width * bmp.height);
                for (let i = 0; i < h.length; i++) h[i] = d[4 * i] * 256 + d[4 * i + 1] + d[4 * i + 2] / 256 - 32768;
                return h;
            }).catch(() => null));
        }
        return cache.get(key);
    };
}

// The tile and pixel under a point at zoom z.
export function locate(lng, lat, zoom) {
    const z = Math.max(5, Math.min(TILE_MAX_ZOOM, Math.round(zoom)));
    const n = 2 ** z, lr = lat * Math.PI / 180;
    const xf = (lng + 180) / 360 * n;
    const yf = (1 - Math.log(Math.tan(lr) + 1 / Math.cos(lr)) / Math.PI) / 2 * n;
    const x = Math.floor(xf), y = Math.floor(yf);
    return {z, x, y, px: Math.min(N - 1, Math.floor((xf - x) * N)), py: Math.min(N - 1, Math.floor((yf - y) * N))};
}

export const pixelSize = (z, y) => {
    const n = 2 ** z, lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 0.5) / n)));
    return 40075016.686 * Math.cos(lat) / (n * N);
};

// Each scale has a light and a dark version: low values sit close to the page colour, high values stand out.
export const DERIVED = {
    slope: {
        label: 'Slope', unit: '°', max: 60, ticks: ['0°', '15°', '30°', '45°', '60°+'],
        light: [[0, '#f4f7fb'], [5, '#dfe8c8'], [12, '#f1e2a0'], [20, '#f2b86c'], [30, '#e07a4f'],
                [40, '#c03a3e'], [50, '#7d1d3a'], [60, '#3d0f2a']],
        dark: [[0, '#16213a'], [8, '#3b1f5c'], [16, '#6b1f6f'], [24, '#a12f5e'], [32, '#d24c3a'],
               [42, '#f07a1f'], [52, '#f9b425'], [60, '#fcf6a3']],
    },
    roughness: {
        label: 'Roughness', unit: 'm', max: 40, ticks: ['0', '10', '20', '30', '40 m+'],
        light: [[0, '#f4f7fb'], [2, '#d3dae6'], [5, '#a9bbdc'], [10, '#6f92cc'], [20, '#2f63b8'],
                [30, '#003893'], [40, '#0a1528']],
        dark: [[0, '#16213a'], [2, '#1d3462'], [5, '#23508f'], [10, '#2f6fc0'], [20, '#5a9be0'],
               [30, '#9cc4f2'], [40, '#e6f0ff']],
    },
    aspect: {label: 'Aspect', unit: '°', flat: 2, flatColor: {light: '#d9dde3', dark: '#3a4458'}},
};

const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));

function ramp(stops, v) {
    if (v <= stops[0][0]) return stops[0][1];
    for (let i = 1; i < stops.length; i++) {
        const [v1, c1] = stops[i];
        if (v <= v1) {
            const [v0, c0] = stops[i - 1], t = (v - v0) / (v1 - v0);
            return c0.map((a, j) => a + (c1[j] - a) * t);
        }
    }
    return stops.at(-1)[1];
}

// Colour lookup tables, so the per-pixel loops only index: slope and roughness every 0.1 unit up to the
// scale's maximum, aspect every degree (index 360 is flat ground).
const STEP = 10;
function table(stops, max) {
    const rgb = stops.map(([v, c]) => [v, hex(c)]), t = new Uint8ClampedArray((max * STEP + 1) * 3);
    for (let i = 0; i <= max * STEP; i++) t.set(ramp(rgb, i / STEP), 3 * i);
    return t;
}
const LUT = Object.fromEntries(['light', 'dark'].map(theme => {
    const aspect = new Uint8ClampedArray(361 * 3);
    for (let d = 0; d < 360; d++) aspect.set(aspectColor(d, theme), 3 * d);
    aspect.set(hex(DERIVED.aspect.flatColor[theme]), 3 * 360);
    return [theme, {slope: table(DERIVED.slope[theme], DERIVED.slope.max),
                    roughness: table(DERIVED.roughness[theme], DERIVED.roughness.max), aspect}];
}));

// HSL to RGB for the aspect wheel: hue = facing direction, so north is red, south is cyan.
export function aspectColor(deg, theme = 'light') {
    const s = 0.55, l = theme === 'dark' ? 0.5 : 0.6, a = s * Math.min(l, 1 - l);
    const f = n => { const k = (n + deg / 30) % 12; return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
    return [f(0), f(8), f(4)];
}

export const compass = deg => ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round(deg / 45) % 8];

// Horn's method on a 3 x 3 window, rows from north to south, cell size in metres.
export function horn([a, b, c, d, , f, g, h, k], cell) {
    const dzdx = ((c + 2 * f + k) - (a + 2 * d + g)) / (8 * cell);
    const dzdy = ((g + 2 * h + k) - (a + 2 * b + c)) / (8 * cell); // positive when height rises southward
    return {slope: Math.atan(Math.hypot(dzdx, dzdy)) * 180 / Math.PI,
            aspect: (Math.atan2(-dzdx, dzdy) * 180 / Math.PI + 360) % 360};
}

const roughness = w => Math.max(...w) - Math.min(...w);

// The tile's heights with a one-pixel border from its neighbours (edge pixels repeat where none exists).
async function padded(tileHeights, z, x, y) {
    const centre = await tileHeights(z, x, y);
    if (!centre) return null;
    const grid = await Promise.all([-1, 0, 1].map(dy => Promise.all([-1, 0, 1].map(dx =>
        dx || dy ? tileHeights(z, x + dx, y + dy) : centre))));
    const W = N + 2, P = new Float32Array(W * W);
    for (let j = -1; j <= N; j++) {
        for (let i = -1; i <= N; i++) {
            const tx = i < 0 ? -1 : i >= N ? 1 : 0, ty = j < 0 ? -1 : j >= N ? 1 : 0;
            let src = grid[ty + 1][tx + 1], ii = i - tx * N, jj = j - ty * N;
            if (!src) { src = centre; ii = Math.min(N - 1, Math.max(0, i)); jj = Math.min(N - 1, Math.max(0, j)); }
            P[(j + 1) * W + i + 1] = src[jj * N + ii];
        }
    }
    return P;
}

// The 3 x 3 window around tile pixel (i, j) in the padded heights, rows from north to south.
function window3(P, i, j) {
    const W = N + 2, r = (j + 1) * W + i + 1;
    return [P[r - W - 1], P[r - W], P[r - W + 1], P[r - 1], P[r], P[r + 1], P[r + W - 1], P[r + W], P[r + W + 1]];
}

// Slope, aspect or roughness as RGBA pixels for one tile, or null where there is no terrain.
// The same Horn and 3 x 3 window as horn() and roughness(), written out so the loop allocates nothing.
export async function renderDerived(tileHeights, kind, z, x, y, theme) {
    const P = await padded(tileHeights, z, x, y);
    if (!P) return null;
    const out = new Uint8ClampedArray(N * N * 4), lut = LUT[theme][kind], W = N + 2, cell8 = 8 * pixelSize(z, y);
    const maxI = (kind === 'slope' ? DERIVED.slope.max : DERIVED.roughness.max) * STEP;
    const deg = 180 / Math.PI, flat = Math.tan(DERIVED.aspect.flat / deg);
    for (let j = 0; j < N; j++) {
        for (let i = 0, r = (j + 1) * W + 1, o = 4 * j * N; i < N; i++, r++, o += 4) {
            const a = P[r - W - 1], b = P[r - W], c = P[r - W + 1], d = P[r - 1], f = P[r + 1],
                  g = P[r + W - 1], h = P[r + W], k = P[r + W + 1];
            let t;
            if (kind === 'roughness') {
                const e = P[r];
                t = Math.min(maxI, Math.round((Math.max(a, b, c, d, e, f, g, h, k) - Math.min(a, b, c, d, e, f, g, h, k)) * STEP));
            } else {
                const dzdx = ((c + 2 * f + k) - (a + 2 * d + g)) / cell8, dzdy = ((g + 2 * h + k) - (a + 2 * b + c)) / cell8;
                const grad = Math.sqrt(dzdx * dzdx + dzdy * dzdy);
                if (kind === 'slope') t = Math.min(maxI, Math.round(Math.atan(grad) * deg * STEP));
                else t = grad < flat ? 360 : Math.round(Math.atan2(-dzdx, dzdy) * deg + 360) % 360;
            }
            out[o] = lut[3 * t]; out[o + 1] = lut[3 * t + 1]; out[o + 2] = lut[3 * t + 2]; out[o + 3] = 255;
        }
    }
    return out;
}

// The measure under a point, from the tile at the given zoom: {slope, aspect} or {roughness}.
export async function derivedAt(tileHeights, kind, lng, lat, zoom) {
    const {z, x, y, px, py} = locate(lng, lat, zoom);
    const P = await padded(tileHeights, z, x, y);
    if (!P) return null;
    const w = window3(P, px, py);
    return kind === 'roughness' ? {roughness: roughness(w)} : horn(w, pixelSize(z, y));
}

// Sun position (degrees) for a time and place, after the NOAA / SunCalc formulas: azimuth clockwise from north.
export function sunPosition(date, lat, lng) {
    const rad = Math.PI / 180, d = date / 86400000 - 0.5 + 2440588 - 2451545, e = rad * 23.4397;
    const M = rad * (357.5291 + 0.98560028 * d);
    const L = M + rad * (1.9148 * Math.sin(M) + 0.02 * Math.sin(2 * M) + 0.0003 * Math.sin(3 * M)) + rad * 102.9372 + Math.PI;
    const dec = Math.asin(Math.sin(e) * Math.sin(L));
    const ra = Math.atan2(Math.sin(L) * Math.cos(e), Math.cos(L));
    const H = rad * (280.16 + 360.9856235 * d) + rad * lng - ra, phi = rad * lat;
    const az = Math.atan2(Math.sin(H), Math.cos(H) * Math.sin(phi) - Math.tan(dec) * Math.cos(phi));
    const alt = Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(H));
    return {azimuth: (az / rad + 180) % 360, altitude: alt / rad};
}

// Heights of a tile and its eight neighbours as one (3 * 256)-pixel square grid (-1e4 where there is no data).
async function grid3(tileHeights, z, x, y) {
    const G = 3 * N, H = new Float32Array(G * G).fill(-1e4);
    const tiles = await Promise.all([-1, 0, 1].flatMap(dy => [-1, 0, 1].map(dx => tileHeights(z, x + dx, y + dy))));
    let top = -1e4;
    tiles.forEach((t, n) => {
        if (!t) return;
        const ox = (n % 3) * N, oy = Math.floor(n / 3) * N;
        for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
            const v = t[j * N + i];
            H[(oy + j) * G + ox + i] = v;
            if (v > top) top = v;
        }
    });
    return {H, G, top};
}

function bilinear(H, G, x, y) {
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, i = y0 * G + x0;
    return (H[i] * (1 - fx) + H[i + 1] * fx) * (1 - fy) + (H[i + G] * (1 - fx) + H[i + G + 1] * fx) * fy;
}

// March from (x, y) towards the sun; 0 = lit, 1 = in shadow, in between near a shadow's edge.
function ray({H, G, top}, x, y, sx, sy, rise, steps, soft) {
    const h0 = bilinear(H, G, x, y) + 1;
    let shade = 0;
    for (let t = 1; t <= steps; t++) {
        const r = h0 + t * rise;
        if (r > top) break;
        const qx = x + t * sx, qy = y + t * sy;
        if (qx < 0 || qy < 0 || qx >= G - 1 || qy >= G - 1) break;
        const above = bilinear(H, G, qx, qy) - r;
        if (above > 0 && (shade = Math.max(shade, above / soft)) >= 1) return 1;
    }
    return shade;
}

// The same march on whole pixels, for the short near pass: no interpolation, so about five times faster.
function nearRay({H, G, top}, x, y, sx, sy, rise, steps, soft) {
    const h0 = H[y * G + x] + 1;
    let shade = 0;
    for (let t = 1; t <= steps; t++) {
        const r = h0 + t * rise;
        if (r > top) break;
        const above = H[Math.round(y + t * sy) * G + Math.round(x + t * sx)] - r;
        if (above > 0 && (shade = Math.max(shade, above / soft)) >= 1) return 1;
    }
    return shade;
}

const SHADOW = [16, 26, 54], SHADOW_ALPHA = 0.5;

// Swiss relief, after the hand-drawn tradition of Imhof and the four ideas behind Eduard
// (https://dilpreet.co/projects/eduard): generalise the terrain, steer the light, use aerial perspective, and
// let the big landforms carry the picture. A plain hillshade reads every 8 m bump at equal strength, which
// turns a Himalayan hillside into noise; this shades three widths of landform and leans on the broadest.
//
//   generalisation   gradients measured over 1, 5 and 15 pixels, weighted 0.2 / 0.35 / 0.45 towards the broad
//   large landforms  the broad gradient also lifts whole sunlit flanks and drops shaded ones
//   ridge emphasis   height above the 15-pixel local mean brightens crests and deepens hollows
//   aerial perspective  low ground loses contrast and cools towards a pale haze, so high ground comes forward
//
// Returned as a see-through overlay drawn over whatever colour layer is beneath, so the page's "colour by
// height" choice decides between grey relief and hypsometric tints.
const SWISS = {
    weight: [0.27, 0.37, 0.36],        // weight on the 1, 5 and 15 pixel gradients
    steps: [1, 5, 15],
    shadow: [44, 56, 78], sun: [255, 253, 243], haze: [186, 201, 217],
    ridge: 0.22,                       // crest brightening
    shadowMax: 0.68, sunMax: 0.5,      // strongest shading either way
    hazeLow: 600, hazeHigh: 4200, hazeMax: 0.1,   // aerial perspective, kept light
};

// Lambert shading of a gradient, as a 0..1 brightness with 0.5 as flat ground.
function lambert(dzdx, dzdn, se, sn, su) {
    const n = Math.hypot(dzdx, dzdn, 1);
    return Math.max(0, (-dzdx * se - dzdn * sn + su) / n);
}

export async function renderSwiss(tileHeights, z, x, y, az, alt) {
    if (!(await tileHeights(z, x, y))) return null;
    const {H, G} = await grid3(tileHeights, z, x, y);
    const px = pixelSize(z, y);
    // A low sun flattens cartographic relief, so the style keeps a plate-like elevation while following the
    // chosen direction. 315 degrees, the Swiss convention, is what the north-west light gives.
    const a = az * Math.PI / 180, e = Math.max(35, alt) * Math.PI / 180;
    const se = Math.sin(a) * Math.cos(e), sn = Math.cos(a) * Math.cos(e), su = Math.sin(e);
    const flat = lambert(0, 0, se, sn, su);
    const wide = SWISS.steps[2];
    const out = new Uint8ClampedArray(N * N * 4);

    for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
            const r = (N + j) * G + N + i;
            let shade = 0;
            for (let k = 0; k < 3; k++) {
                const d = SWISS.steps[k], span = 2 * d * px;
                const dzdx = (H[r + d] - H[r - d]) / span;
                const dzdn = -(H[r + d * G] - H[r - d * G]) / span;  // grid rows run south
                shade += SWISS.weight[k] * lambert(dzdx, dzdn, se, sn, su);
            }
            // Lambert leaves far more room below flat ground than above it, so stretch each side to -1..+1
            // before shading; otherwise the shadows swamp the highlights.
            let lit = shade >= flat ? (shade - flat) / (1 - flat) : (shade - flat) / flat;

            // Height above the local mean: positive on a crest, negative in a hollow.
            const mean = (H[r - wide] + H[r + wide] + H[r - wide * G] + H[r + wide * G]) / 4;
            lit += SWISS.ridge * Math.max(-1, Math.min(1, (H[r] - mean) / (0.35 * wide * px)));
            lit = Math.max(-1, Math.min(1, lit));

            // Aerial perspective: the lowest ground loses a little contrast and takes on haze, so the high
            // country comes forward. Imhof's version is subtle, and so is this.
            const t = Math.max(0, Math.min(1, (H[r] - SWISS.hazeLow) / (SWISS.hazeHigh - SWISS.hazeLow)));
            const haze = SWISS.hazeMax * (1 - t);
            lit *= 1 - 0.2 * (1 - t);

            const o = 4 * (j * N + i);
            const [tint, strength] = lit < 0 ? [SWISS.shadow, SWISS.shadowMax * (-lit) ** 0.85]
                                             : [SWISS.sun, SWISS.sunMax * lit ** 0.85];
            // Blend the haze over the shading: both are see-through, so mix them by weight.
            const total = strength + haze;
            if (total <= 0) { out[o + 3] = 0; continue; }
            for (let c = 0; c < 3; c++) out[o + c] = (tint[c] * strength + SWISS.haze[c] * haze) / total;
            out[o + 3] = 255 * Math.min(0.8, total);
        }
    }
    return out;
}

// Cast shadow for one tile as RGBA pixels, in two passes.
// Far: two zooms coarser, over the coarse tile and its eight neighbours, so ridges up to one coarse tile away
// (about 35 km at zoom 12) can shade it. Long shadows are smooth, so it is traced on every fourth coarse pixel
// and blended; it is the same for the 16 map tiles under one coarse tile, so it is kept per coarse tile and sun.
// Near: at the tile's own resolution, 64 pixels towards the sun, for crisp shadows from nearby ridges; traced on
// every second pixel and blended in between.
// The slope's own light and dark side comes from MapLibre's hillshade, lit from the same sun.
const farMasks = new Map();

const FAR = 4, FS = N / FAR + 3;  // far mask: one sample every 4 coarse pixels, from -4 to 260

function farMask(coarse, key, sx, sy, rise) {
    if (!farMasks.has(key)) {
        if (farMasks.size > 24) farMasks.delete(farMasks.keys().next().value);
        const m = new Float32Array(FS * FS), soft = Math.max(4, 2 * rise);
        for (let j = 0; j < FS; j++) {
            for (let i = 0; i < FS; i++) m[j * FS + i] = ray(coarse, N + FAR * (i - 1), N + FAR * (j - 1), sx, sy, rise, coarse.G, soft);
        }
        farMasks.set(key, m);
    }
    return farMasks.get(key);
}

export async function renderShadow(tileHeights, z, x, y, az, alt) {
    if (alt <= 0 || !(await tileHeights(z, x, y))) return null;
    const sx = Math.sin(az * Math.PI / 180), sy = -Math.cos(az * Math.PI / 180);  // towards the sun; grid y runs south
    const tan = Math.tan(alt * Math.PI / 180);
    const zc = Math.max(5, z - 2), k = 2 ** (z - zc), cx = x >> (z - zc), cy = y >> (z - zc);
    const [fine, coarse] = await Promise.all([grid3(tileHeights, z, x, y), grid3(tileHeights, zc, cx, cy)]);
    const fineRise = tan * pixelSize(z, y);
    const far = farMask(coarse, `${zc}/${cx}/${cy}/${az}/${alt}`, sx, sy, tan * pixelSize(zc, cy));
    const ox = (x - cx * k) * (N / k), oy = (y - cy * k) * (N / k);  // this tile's corner, in coarse pixels

    const H2 = N / 2 + 1, near = new Float32Array(H2 * H2), soft = Math.max(3, 2 * fineRise);
    for (let j = 0; j < H2; j++) {
        for (let i = 0; i < H2; i++) near[j * H2 + i] = nearRay(fine, N + Math.min(N - 1, 2 * i), N + Math.min(N - 1, 2 * j), sx, sy, fineRise, 64, soft);
    }
    const out = new Uint8ClampedArray(N * N * 4);
    for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
            let s = Math.min(1, bilinear(near, H2, Math.min(H2 - 1.001, i / 2), Math.min(H2 - 1.001, j / 2)));
            if (s < 1) s = Math.max(s, Math.min(1, bilinear(far, FS, (ox + (i + 0.5) / k - 0.5) / FAR + 1,
                                                                        (oy + (j + 0.5) / k - 0.5) / FAR + 1)));
            const o = 4 * (j * N + i);
            out[o] = SHADOW[0]; out[o + 1] = SHADOW[1]; out[o + 2] = SHADOW[2]; out[o + 3] = 255 * SHADOW_ALPHA * s;
        }
    }
    return out;
}
