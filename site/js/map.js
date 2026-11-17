import {COG_URL, VERSION, VERSIONS, MAX_BOUNDS, NEPAL_BOUNDS, THEMES, TILE_MAX_ZOOM, applyTheme, elevationAt, fmtM, inNepal, loadNeighbours, loadNepal,
    redrawDrape, reliefStops, terrainStyle} from './terrain.js';
import {PALETTES, rememberPalette, stopsFor, storedPalette} from './palettes.js';
import {DERIVED, SHADE_ALT, aspectColor, compass, derivedAt, derivedSource, horn, sunPosition} from './derive.js';
import {currentTheme, initThemeToggle} from './theme.js';

const $ = id => document.getElementById(id);

// Places worth flying to, by province, west to east. Each carries a camera: the mountains are worth seeing
// from low down, the Terai from higher up. A peak on the border is viewed from a point inside Nepal looking
// towards it, because the ground on the far side is not in the model and would leave half the screen empty.
const PLACES = {
    Sudurpashchim: [
        ['Api Himal', 80.92, 29.94, 11.5, 10, 70],
        ['Khaptad', 81.12, 29.36, 11, 0, 55],
        ['Bheemdatt', 80.18, 28.97, 11.5, 0, 45],
        ['Dhangadhi', 80.58, 28.70, 11, 0, 40],
    ],
    Karnali: [
        ['Rara Lake', 82.09, 29.53, 12, 0, 60],
        ['Shey Phoksundo', 82.95, 29.19, 11.5, -20, 68],
        ['Jumla', 82.18, 29.27, 11, 0, 55],
        ['Dolpa', 82.82, 28.98, 10.8, 0, 60],
    ],
    Lumbini: [
        ['Lumbini', 83.28, 27.48, 12, 0, 35],
        ['Palpa (Tansen)', 83.55, 27.87, 11.5, 0, 55],
        ['Butwal', 83.46, 27.70, 11.5, 0, 45],
    ],
    Gandaki: [
        ['Annapurna', 83.82, 28.596, 11.2, 10, 68],
        ['Pokhara', 83.98, 28.23, 11, 0, 62],
        ['Manaslu', 84.56, 28.55, 11.3, -15, 70],
        ['Upper Mustang', 83.87, 29.10, 10.8, 0, 60],
        ['Muktinath', 83.87, 28.82, 11.5, 15, 65],
    ],
    Bagmati: [
        ['Kathmandu Valley', 85.32, 27.70, 10.8, -10, 55],
        ['Langtang', 85.55, 28.21, 11.3, 20, 68],
        ['Gosaikunda', 85.42, 28.08, 12, 0, 62],
        ['Chitwan', 84.40, 27.58, 10.5, 0, 45],
    ],
    Madhesh: [
        ['Janakpur', 85.92, 26.73, 11.5, 0, 35],
        ['Birgunj', 84.88, 27.01, 11.5, 0, 35],
    ],
    Koshi: [
        ['Everest', 86.88, 27.94, 11.8, 30, 70],
        ['Kanchenjunga', 88.07, 27.65, 11.3, 40, 68],
        ['Makalu', 87.05, 27.84, 11.3, 35, 70],
        ['Ilam', 87.93, 26.91, 11.5, 0, 50],
        ['Dharan', 87.28, 26.81, 11.5, 0, 40],
    ],
};

// Visualization types. Hillshade can be coloured by height. Slope, aspect and roughness are computed from the
// terrain tiles in workers (derive.js) and drawn under a soft hillshade so the landform still reads.
const VIZ = {
    'hillshade': {label: 'Hillshade', shade: 'full'},
    'swiss': {label: 'Swiss relief', swiss: true},
    'color-relief': {label: 'Color relief', relief: true},
    'slope': {label: 'Slope', derived: 'slope', shade: 'soft'},
    'aspect': {label: 'Aspect', derived: 'aspect', shade: 'soft'},
    'roughness': {label: 'Roughness', derived: 'roughness', shade: 'soft'},
};
const DATA_BOUNDS = [79.79, 26.08, 88.3, 30.81];
const params = new URLSearchParams(location.search);
let viz = VIZ[params.get('viz')] ? params.get('viz') : 'hillshade';
let colorOn = params.get('color') !== '0';  // hillshade coloured by height (the old "color-hillshade" view)
let light = 'nw';      // 'nw': a fixed direction, below; 'sun': the real sun for a date and time, with cast shadows
let fixedAz = 315;     // the Swiss convention, from the north-west

let palette = storedPalette();

const [nepal, neighbours] = await Promise.all([loadNepal(), loadNeighbours()]);
const style = terrainStyle(nepal, {theme: THEMES[currentTheme()], palette, mode: currentTheme(), neighbours});
const SHADE = {
    full: style.layers.find(l => l.id === 'hillshade').paint['hillshade-exaggeration'],
    soft: ['interpolate', ['linear'], ['zoom'], 5, 0.4, 12, 0.3, 15, 0.2],
    sun: ['interpolate', ['linear'], ['zoom'], 5, 0.45, 12, 0.35, 15, 0.25],  // lighter, so cast shadows stand out
};
const derivedQuery = () => `?t=${currentTheme()}&p=${palette}`;
for (const k of Object.keys(DERIVED)) style.sources[`derive-${k}`] = derivedSource(k, DATA_BOUNDS, derivedQuery());
// The land under the shading: pale paper in light mode, slate in dark mode.
const LAND = {light: '#eef0f2', dark: '#3a4458'};
style.layers.find(l => l.id === 'land').paint['fill-color'] = LAND[currentTheme()];
style.layers.splice(style.layers.findIndex(l => l.id === 'hillshade'), 0, ...Object.keys(DERIVED).map(k =>
    ({id: k, type: 'raster', source: `derive-${k}`, paint: {'raster-fade-duration': 0}})));
// Real sun: MapLibre's hillshade turns to the sun on the GPU while the time slider moves; cast shadows from the
// workers fill in when it is let go. At night a dark veil covers the country.
style.sources['derive-shadow'] = derivedSource('shadow', DATA_BOUNDS, '?az=180&alt=45');
style.sources['derive-swiss'] = derivedSource('swiss', DATA_BOUNDS, '?az=315&alt=45');
style.layers.splice(style.layers.findIndex(l => l.id === 'mask'), 0,
    {id: 'swiss', type: 'raster', source: 'derive-swiss', layout: {visibility: 'none'}, paint: {'raster-fade-duration': 0}},
    {id: 'shadow', type: 'raster', source: 'derive-shadow', layout: {visibility: 'none'}, paint: {'raster-fade-duration': 0}},
    {id: 'night', type: 'fill', source: 'nepal', layout: {visibility: 'none'}, paint: {'fill-color': '#0a1528', 'fill-opacity': 0.55}});

// Summits found in NepalDTM (scripts/summits.py), loaded when first shown. Bigger dots stand out more
// from their surroundings (higher prominence); names come from OpenStreetMap where a mapped peak is close.
style.glyphs = 'https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf';
// The neighbouring countries in the frame get their names, so the faint land around Nepal reads as
// somewhere rather than as a smudge. Only this page has glyphs; the home page draws the land unlabelled.
if (neighbours) style.layers.splice(style.layers.findIndex(l => l.id === 'border'), 0,
    {id: 'neighbours-label', type: 'symbol', source: 'neighbours', filter: ['==', ['get', 'kind'], 'label'],
     layout: {'text-field': ['upcase', ['get', 'name']], 'text-font': ['Noto Sans Medium'], 'text-size': 13,
              'text-letter-spacing': 0.22},
     paint: {'text-color': THEMES[currentTheme()].edge, 'text-opacity': 0.5}});
style.sources.peaks = {type: 'geojson', data: {type: 'FeatureCollection', features: []},
    attribution: 'Peak names © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'};
const NAME = ['coalesce', ['get', 'name'], ''];
// Height shown and filtered on: the surveyed (OpenStreetMap) height where there is one, otherwise NepalDTM's,
// marked "≈" because sharp summits read low in an 8 m model.
const HEIGHT = ['coalesce', ['get', 'osm'], ['get', 'h']];
const HEIGHT_TEXT = ['concat', ['case', ['has', 'osm'], '', '≈ '], ['number-format', HEIGHT, {locale: 'en-US', 'max-fraction-digits': 0}], ' m'];
style.layers.push(
    {id: 'peaks-dot', type: 'circle', source: 'peaks', layout: {visibility: 'none', 'circle-sort-key': ['get', 'prom']},
     paint: {'circle-radius': ['interpolate', ['linear'], ['get', 'prom'], 100, 2.5, 500, 4, 2000, 6.5], 'circle-color': '#dc143c',
             'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.5}},
    {id: 'peaks-label', type: 'symbol', source: 'peaks', layout: {
        visibility: 'none', 'symbol-sort-key': ['-', 0, ['+', ['get', 'prom'], ['/', ['get', 'h'], 10000]]],
        'text-font': ['Noto Sans Medium'], 'text-size': 12,
        'text-field': ['format', NAME, {}, ['case', ['==', NAME, ''], '', '\n'], {}, HEIGHT_TEXT, {'font-scale': 0.85}],
        'text-anchor': 'top', 'text-offset': [0, 0.6], 'text-max-width': 9},
     paint: {'text-color': '#101d36', 'text-halo-color': 'rgba(255,255,255,.92)', 'text-halo-width': 1.4}},
);

// A degree grid, off until the button under the controls is pressed. One line every degree across the
// frame, with its value repeated along it so a reading is never far from wherever you are looking.
const GRID_INK = {light: ['#101d36', 'rgba(255,255,255,.85)'], dark: ['#c3cedf', 'rgba(10,21,40,.85)']};
const round6 = v => Math.round(v * 1e6) / 1e6;

// Lines are drawn in short hops rather than as two points, so they follow the ground in 3D.
function hop(from, to, step = 0.1) {
    const out = [];
    for (let v = from; v < to; v += step) out.push(round6(v));
    out.push(round6(to));
    return out;
}

function graticule([[west, south], [east, north]], step = 1) {
    const round = round6;
    const marks = (from, to) => {
        const out = [];
        for (let v = Math.ceil(from / step) * step; v <= to; v += step) out.push(round(v));
        return out;
    };
    const line = (coordinates, label) => ({type: 'Feature', properties: {label}, geometry: {type: 'LineString', coordinates}});
    return {type: 'FeatureCollection', features: [
        ...marks(west, east).map(lng => line(hop(south, north).map(lat => [lng, lat]), `${Math.abs(lng)}°${lng < 0 ? 'W' : 'E'}`)),
        ...marks(south, north).map(lat => line(hop(west, east).map(lng => [lng, lat]), `${Math.abs(lat)}°${lat < 0 ? 'S' : 'N'}`)),
    ]};
}
style.sources.grid = {type: 'geojson', data: graticule(MAX_BOUNDS)};
style.layers.push(
    {id: 'grid', type: 'line', source: 'grid', layout: {visibility: 'none'},
     paint: {'line-color': GRID_INK[currentTheme()][0], 'line-opacity': 0.35, 'line-width': 0.8,
             'line-dasharray': [2, 3]}},
    {id: 'grid-label', type: 'symbol', source: 'grid', layout: {
        visibility: 'none', 'symbol-placement': 'line', 'symbol-spacing': 280, 'text-field': ['get', 'label'],
        'text-font': ['Noto Sans Medium'], 'text-size': 10.5, 'text-letter-spacing': 0.05, 'text-padding': 4},
     paint: {'text-color': GRID_INK[currentTheme()][0], 'text-halo-color': GRID_INK[currentTheme()][1],
             'text-halo-width': 1.2}},
);

// With the grid on, the pointer drags a meridian and a parallel of its own across the map, labelled with
// where it is. The lines are geographic rather than drawn across the screen, so they stay true when the
// map is turned or tilted.
const NOTHING = {type: 'FeatureCollection', features: []};
style.sources.crosshair = {type: 'geojson', data: NOTHING};
style.layers.push(
    {id: 'crosshair', type: 'line', source: 'crosshair', filter: ['==', ['geometry-type'], 'LineString'],
     layout: {visibility: 'none'},
     paint: {'line-color': GRID_INK[currentTheme()][0], 'line-opacity': 0.8, 'line-width': 1.1}},
    {id: 'crosshair-label', type: 'symbol', source: 'crosshair', filter: ['==', ['geometry-type'], 'Point'],
     layout: {visibility: 'none', 'text-field': ['get', 'label'], 'text-font': ['Noto Sans Medium'],
              'text-size': 11.5, 'text-anchor': 'top-left', 'text-offset': [0.7, 0.5],
              'text-allow-overlap': true, 'text-ignore-placement': true},
     paint: {'text-color': GRID_INK[currentTheme()][0], 'text-halo-color': GRID_INK[currentTheme()][1],
             'text-halo-width': 1.4}},
);

for (const [layer, value] of Object.entries(vizLayers(viz))) {
    const l = style.layers.find(x => x.id === layer);
    l.layout = {...l.layout, visibility: value};
}
style.layers.find(l => l.id === 'hillshade').paint['hillshade-exaggeration'] = shadeOf(viz);
style.sources.profile = {type: 'geojson', data: {type: 'FeatureCollection', features: []}};
style.layers.push(
    {id: 'profile-casing', type: 'line', source: 'profile', paint: {'line-color': '#ffffff', 'line-width': 6, 'line-opacity': 0.9}},
    {id: 'profile-line', type: 'line', source: 'profile', paint: {'line-color': '#dc143c', 'line-width': 2.5}},
);

// The panel and the controls sit over the canvas, so the whole country is framed into what is left of
// it: otherwise the far west starts out behind the panel.
const CHROME = innerWidth > 760
    ? {top: 76, bottom: 48, left: 292, right: 70}   // panel on the left, map controls on the right
    : {top: 64, bottom: 96, left: 16, right: 16};   // the panel is a sheet along the bottom
// The padding goes into the camera, not onto the map: padding left on the map is also counted against
// maxBounds, which would stop it zooming out far enough to show the country it has just been asked for.
const allOfNepalCamera = () => map.cameraForBounds(NEPAL_BOUNDS, {padding: CHROME});
// Read before the map opens: with hash: true it writes its own position there as soon as it has one.
const SHARED_VIEW = location.hash;

// The viewer opens flat. The terrain is still in the style's sources, so the 3D button, the tilt slider
// and a shared link that was tilted can all put it back under the map.
delete style.terrain;

const map = new maplibregl.Map({
    container: 'map',
    style,
    bounds: NEPAL_BOUNDS,
    pitch: 0,
    maxBounds: MAX_BOUNDS,
    minZoom: 5.5,
    maxZoom: 22,  // past 17 (G) or 14 (v1, v2) this magnifies; it adds no detail
    maxPitch: 85,
    hash: true,
    attributionControl: false,
});
// A link with a position in it says where to go; otherwise open on the whole country, clear of the panel.
if (!SHARED_VIEW) map.jumpTo(allOfNepalCamera());
map.addControl(new maplibregl.AttributionControl({
    compact: true,
    customAttribution: 'CC BY 4.0',
}));
map.addControl(new maplibregl.NavigationControl({visualizePitch: true}), 'top-right');
map.addControl(new maplibregl.FullscreenControl(), 'top-right');
// The grid button sits with the other map controls rather than in the panel, because it is about
// reading the map rather than about what is on it.
class GridControl {
    onAdd() {
        this.container = document.createElement('div');
        this.container.className = 'maplibregl-ctrl maplibregl-ctrl-group';
        this.container.innerHTML = `<button type="button" class="grid-btn" aria-pressed="false"
            title="Coordinate grid" aria-label="Coordinate grid"><svg viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="1.6"><path d="M3 3h18v18H3z"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>
            </svg></button>`;
        this.button = this.container.firstElementChild;
        this.button.onclick = () => showGrid(this.button.getAttribute('aria-pressed') === 'false');
        return this.container;
    }

    onRemove() {
        this.container.remove();
    }
}
const gridControl = new GridControl();
map.addControl(gridControl, 'top-right');
map.addControl(new maplibregl.ScaleControl({maxWidth: 120}), 'bottom-right');

let gridOn = false;

function showGrid(on) {
    gridOn = on;
    gridControl.button.setAttribute('aria-pressed', String(on));
    for (const id of ['grid', 'grid-label', 'crosshair', 'crosshair-label']) {
        map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
    }
    if (!on) map.getSource('crosshair').setData(NOTHING);
    restCursor();
}

// The pointer is a target while the grid is on and a crosshair while the profile tool is drawing a line;
// over a summit it is a hand.
function restCursor() {
    map.getCanvas().style.cursor = profiling ? 'crosshair' : gridOn ? 'var(--cursor-target)' : '';
}

// One meridian and one parallel through the pointer, and its position written beside it. The redraw
// waits for the next frame, so a fast drag across the map costs one update per frame rather than per
// mouse event.
const degrees = (v, pos, neg) => `${Math.abs(v).toFixed(4)}°${v < 0 ? neg : pos}`;
let crosshairAt = null;
let crosshairQueued = false;

function drawCrosshair() {
    crosshairQueued = false;
    if (!gridOn || !crosshairAt) return;
    const {lng, lat} = crosshairAt;
    const [[west, south], [east, north]] = MAX_BOUNDS;
    const lngText = degrees(lng, 'E', 'W');
    const latText = degrees(lat, 'N', 'S');
    map.getSource('crosshair').setData({type: 'FeatureCollection', features: [
        {type: 'Feature', properties: {}, geometry: {type: 'LineString',
         coordinates: hop(south, north).map(y => [round6(lng), y])}},
        {type: 'Feature', properties: {}, geometry: {type: 'LineString',
         coordinates: hop(west, east).map(x => [x, round6(lat)])}},
        {type: 'Feature', properties: {label: `${latText}\n${lngText}`},
         geometry: {type: 'Point', coordinates: [round6(lng), round6(lat)]}},
    ]});
}

map.on('mousemove', e => {
    if (!gridOn) return;
    crosshairAt = e.lngLat;
    if (crosshairQueued) return;
    crosshairQueued = true;
    requestAnimationFrame(drawCrosshair);
});
map.on('mouseout', () => {
    if (!gridOn) return;
    crosshairAt = null;
    map.getSource('crosshair').setData(NOTHING);
});

// View and layer controls

const PEAK_TEXT = {light: ['#101d36', 'rgba(255,255,255,.92)'], dark: ['#f4f7fb', 'rgba(10,21,40,.9)']};
function peakTheme(theme) {
    map.setPaintProperty('peaks-label', 'text-color', PEAK_TEXT[theme][0]);
    map.setPaintProperty('peaks-label', 'text-halo-color', PEAK_TEXT[theme][1]);
}
map.once('style.load', () => {
    peakTheme(currentTheme());
    applyPalette();  // a scheme remembered from last time also decides how the hillshade is lit
    if (map.getPitch() > 0) setMode(true, false);  // a link shared from a tilted view comes back tilted
});
function gridTheme(theme) {
    const [ink, halo] = GRID_INK[theme];
    for (const id of ['grid', 'crosshair']) map.setPaintProperty(id, 'line-color', ink);
    for (const id of ['grid-label', 'crosshair-label']) {
        map.setPaintProperty(id, 'text-color', ink);
        map.setPaintProperty(id, 'text-halo-color', halo);
    }
}
initThemeToggle(theme => {
    peakTheme(theme);
    gridTheme(theme);
    map.setPaintProperty('land', 'fill-color', LAND[theme]);
    applyPalette();
    applyTheme(map, THEMES[theme]);
    showViz();
});

// The hillshade is warm by default, after the Swiss convention of a yellow sun on blue shadow. Under the
// greyscale scheme that warmth is the only hue left on the map, so there it goes neutral too.
const SHADE_COLOUR = {
    tinted: ['rgb(64, 82, 104)', 'rgb(255, 249, 214)', 'rgb(88, 102, 118)'],
    grey: ['rgb(72, 76, 82)', 'rgb(250, 250, 249)', 'rgb(98, 102, 108)'],
};

// Swap every ramp: the height colours are a paint property, the measures are baked into their tiles.
function applyPalette() {
    const theme = currentTheme();
    const [shadow, highlight, accent] = SHADE_COLOUR[palette === 'grey' ? 'grey' : 'tinted'];
    map.setPaintProperty('hillshade', 'hillshade-shadow-color', shadow);
    map.setPaintProperty('hillshade', 'hillshade-highlight-color', highlight);
    map.setPaintProperty('hillshade', 'hillshade-accent-color', accent);
    map.setPaintProperty('relief', 'color-relief-color',
                         ['interpolate', ['linear'], ['elevation'], ...reliefStops(palette, theme).flat()]);
    for (const k of Object.keys(DERIVED)) {
        map.getSource(`derive-${k}`).setTiles([`derive://${k}/{z}/{x}/{y}${derivedQuery()}`]);
    }
    redrawDrape(map);
}

let exaggeration = 1.5;

function setMode(three, pitch) {
    $('mode-3d').setAttribute('aria-pressed', three);
    $('mode-2d').setAttribute('aria-pressed', !three);
    map.setTerrain(three ? {source: 'terrain', exaggeration} : null);
    if (pitch !== false) map.easeTo({pitch: three ? Math.max(map.getPitch(), 55) : 0, duration: 800});
}

$('mode-2d').onclick = () => setMode(false);
$('mode-3d').onclick = () => setMode(true);
$('exag').oninput = e => {
    exaggeration = parseFloat(e.target.value);
    $('exag-val').textContent = `${exaggeration.toFixed(1)}×`;
    if (map.getTerrain()) map.setTerrain({source: 'terrain', exaggeration});
};
$('lyr-border').onchange = e => map.setLayoutProperty('border', 'visibility', e.target.checked ? 'visible' : 'none');

// Rotation: a slider, quarter turns, north up, and a slow spin that stops when you touch the map.

const signed = b => ((b + 540) % 360) - 180;
function showBearing() {
    const b = Math.round(signed(map.getBearing()));
    $('bearing').value = b;
    $('bearing-val').textContent = `${b}°`;
}
function showPitch() {
    const p = Math.round(map.getPitch());
    $('pitch').value = p;
    $('pitch-val').textContent = `${p}°`;
}
let spinning = false;
function stopSpin() {
    if (!spinning) return;
    spinning = false;
    $('rot-spin').setAttribute('aria-pressed', false);
    map.stop();
}
function spinStep() {
    if (spinning) map.easeTo({bearing: map.getBearing() + 45, duration: 9000, easing: t => t});
}
map.on('rotate', showBearing);
map.on('pitch', showPitch);
map.on('moveend', spinStep);
showBearing();
showPitch();
$('bearing').oninput = e => { stopSpin(); map.setBearing(+e.target.value); };
// Tilting needs the terrain on, so dragging the slider up switches to 3D.
$('pitch').oninput = e => {
    const p = +e.target.value;
    if (p > 0 && !map.getTerrain()) setMode(true, false);
    map.setPitch(p);
};
$('rot-left').onclick = () => { stopSpin(); map.easeTo({bearing: map.getBearing() - 90, duration: 900}); };
$('rot-right').onclick = () => { stopSpin(); map.easeTo({bearing: map.getBearing() + 90, duration: 900}); };
$('rot-north').onclick = () => { stopSpin(); map.easeTo({bearing: 0, duration: 900}); };
$('tilt-flat').onclick = () => setMode(false);
for (const [id, deg] of [['tilt-45', 45], ['tilt-65', 65], ['tilt-max', 85]]) {
    $(id).onclick = () => {
        if (!map.getTerrain()) setMode(true, false);
        map.easeTo({pitch: deg, duration: 700});
    };
}
$('rot-spin').onclick = () => {
    if (spinning) return stopSpin();
    spinning = true;
    $('rot-spin').setAttribute('aria-pressed', true);
    spinStep();
};
for (const ev of ['mousedown', 'touchstart', 'wheel']) map.getCanvas().addEventListener(ev, stopSpin, {passive: true});

// The View and Light popups. Each control lives in one place only, so there is nothing to keep in step, and
// opening one closes the other.
const POPUPS = {view: 'view-popup', sun: 'sun-popup', places: 'places-popup'};

function showPopup(which) {
    for (const [name, id] of Object.entries(POPUPS)) {
        $(id).hidden = name !== which;
        $(`${name}-toggle`).setAttribute('aria-expanded', name === which);
    }
}

for (const [name, id] of Object.entries(POPUPS)) {
    $(`${name}-toggle`).onclick = e => {
        e.stopPropagation();
        showPopup($(id).hidden ? name : null);
    };
    $(id).onclick = e => e.stopPropagation();
}
document.addEventListener('click', () => showPopup(null));
document.addEventListener('keydown', e => { if (e.key === 'Escape') showPopup(null); });

// Visualization picker and legend

function vizLayers(name) {
    const v = VIZ[name], on = x => x ? 'visible' : 'none';
    const coloured = (name === 'hillshade' || name === 'swiss') && colorOn;
    return {relief: on(v.relief || coloured), swiss: on(v.swiss),
            hillshade: on((v.shade || light === 'sun') && !v.swiss),
            ...Object.fromEntries(Object.keys(DERIVED).map(k => [k, on(v.derived === k)]))};
}
function shadeOf(name) { return SHADE[VIZ[name].shade === 'soft' ? 'soft' : light === 'sun' ? 'sun' : 'full']; }
function aboutKey(name) { return name === 'hillshade' && colorOn ? 'color-hillshade' : name; }

const gradient = (stops, max) => `linear-gradient(90deg, ${stops.map(([v, c]) => `${c} ${(v / max * 100).toFixed(1)}%`).join(', ')})`;
const wheel = theme => `conic-gradient(${Array.from({length: 13}, (_, i) =>
    `rgb(${aspectColor((i * 30) % 360, theme).map(Math.round)}) ${i * 30}deg`).join(', ')})`;
const diagonal = g => g.replace('90deg', '135deg');

function swatch(name, theme) {
    const relief = diagonal(gradient(reliefStops(palette, theme), 8850));
    return {
        'hillshade': theme === 'dark' ? 'linear-gradient(135deg, #c9d2df, #5d6b80 55%, #1d2638)'
                                      : 'linear-gradient(135deg, #fffbe6, #9aa6b4 55%, #4a5a6c)',
        'swiss': theme === 'dark'
            ? 'linear-gradient(135deg, #e8eef6, #8ea3bd 45%, #2a3446)'
            : 'linear-gradient(135deg, #fffcee, #b0c2d6 45%, #263040)',
        'color-relief': relief,
        'color-hillshade': `linear-gradient(135deg, rgba(255,251,230,.5), rgba(64,82,104,.55)), ${relief}`,
        'slope': diagonal(gradient(stopsFor(palette, theme, 'slope'), DERIVED.slope.max)),
        'aspect': wheel(theme),
        'roughness': diagonal(gradient(stopsFor(palette, theme, 'roughness'), DERIVED.roughness.max)),
    }[name];
}

// What each view shows, in plain words, under its legend.
const ABOUT = {
    'hillshade': 'Shading as if the sun stood in the north-west, so ridges, valleys and terraces stand out. ' +
        'It shows the shape of the land, not its height.',
    'swiss': 'Shaded relief in the Swiss tradition of Eduard Imhof. A plain hillshade treats every 8 m bump alike, ' +
        'which turns a steep hillside into noise; this shades three widths of landform and leans on the broadest, so ' +
        'ridges and valleys read as whole shapes. Crests are brightened, hollows deepened, and low ground fades into a ' +
        'pale haze so the high country comes forward. Turn off "Colour by height" for grey relief alone.',
    'color-relief': 'Colour by height above sea level, with no shading at all, so heights are easy to compare ' +
        'and nothing is lost in a shadow.',
    'color-hillshade': 'Height colours with hillshade on top: the colour tells you how high the ground is, ' +
        'the shading shows its shape.',
    'slope': 'How steep the ground is, in degrees from flat. Low values are flat fields and valley floors; high values are ' +
        'steep hillsides. Slopes over about 30° are where landslides and avalanches start, and on slopes of 40° and more ' +
        'NepalDTM is least accurate (about 11 m RMSE).',
    'aspect': 'Which way each slope faces. North-facing slopes (red) get the least sun in Nepal and usually stay cooler and ' +
        'moister; south-facing slopes (cyan) get the most, warm up and dry out sooner, and lose their snow first.',
    'roughness': 'How uneven the ground is: the largest height difference between a pixel and its eight neighbours. ' +
        'Low values are smooth (river terraces, the Terai plain); high values are rugged (cliffs, gullies, moraines). ' +
        'Areas filled from 30 m data look smoother than they really are.',
};
const ZOOM_NOTE = '<p class="legend-note">Worked out from the map tiles at the current zoom; finest at zoom 14 (8.4 m). Lower zooms read gentler.</p>';

function legendHtml(name, theme) {
    const v = VIZ[name];
    const ramped = v.relief || v.derived === 'slope' || v.derived === 'roughness'
        || ((name === 'hillshade' || name === 'swiss') && colorOn);
    const about = `<p class="legend-about">${ABOUT[aboutKey(name)]}` +
        (ramped ? ` <span class="legend-scheme">${PALETTES[palette].note}</span>` : '') + '</p>';
    const bar = (bg, ticks) => `<div class="legend-bar" style="background:${bg}"></div>` +
        `<div class="legend-ticks">${ticks.map(t => `<span>${t}</span>`).join('')}</div>`;
    if ((name === 'hillshade' || name === 'swiss') && !colorOn) return about;
    if (v.relief || name === 'hillshade' || name === 'swiss') {
        return bar(gradient(reliefStops(palette, theme), 8850), ['0', '2,000', '4,000', '6,000', '8,850 m']) + about;
    }
    if (v.derived === 'aspect') {
        return `<div class="legend-wheel"><div class="wheel" style="background:${wheel(theme)}">` +
            '<span class="n">N</span><span class="e">E</span><span class="s">S</span><span class="w">W</span></div>' +
            `<p><i class="flat" style="background:${DERIVED.aspect.flatColor[theme]}"></i> Flat ground (under 2°).</p></div>` +
            about + ZOOM_NOTE;
    }
    const d = DERIVED[v.derived];
    return bar(gradient(stopsFor(palette, theme, v.derived), d.max), d.ticks) + about + ZOOM_NOTE;
}

for (const [name, v] of Object.entries(VIZ)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.dataset.viz = name;
    b.innerHTML = `<i class="swatch"></i>${v.label}`;
    b.onclick = () => setViz(name);
    $('viz').append(b);
}

function setViz(name) {
    viz = name;
    for (const [layer, value] of Object.entries(vizLayers(name))) map.setLayoutProperty(layer, 'visibility', value);
    map.setPaintProperty('hillshade', 'hillshade-exaggeration', shadeOf(name));
    redrawDrape(map);
    showViz();
}

function showViz() {
    const theme = currentTheme();
    for (const b of $('viz').children) {
        b.setAttribute('aria-checked', b.dataset.viz === viz);
        b.firstElementChild.style.background = swatch(b.dataset.viz === 'hillshade' && colorOn ? 'color-hillshade' : b.dataset.viz, theme);
    }
    $('color-wrap').hidden = viz !== 'hillshade' && viz !== 'swiss';
    $('viz-color').checked = colorOn;
    $('legend').innerHTML = legendHtml(viz, theme);
    showPalette();
    const url = new URL(location.href);
    url.searchParams.delete('viz');
    url.searchParams.delete('color');
    if (viz !== 'hillshade') url.searchParams.set('viz', viz);
    else if (!colorOn) url.searchParams.set('color', '0');
    history.replaceState(history.state, '', url);
}
$('viz-color').onchange = e => { colorOn = e.target.checked; setViz(viz); };

// Colour scheme. Only the ramped layers use it, so the row is hidden for plain hillshade.
for (const [name, p] of Object.entries(PALETTES)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.dataset.palette = name;
    b.title = `${p.label} — ${p.note}`;
    b.innerHTML = `<i class="swatch"></i>${p.label}`;
    b.onclick = () => {
        palette = name;
        rememberPalette(name);
        applyPalette();
        showViz();
    };
    $('palette').append(b);
}

function showPalette() {
    const theme = currentTheme();
    const ramped = VIZ[viz].relief || VIZ[viz].derived === 'slope' || VIZ[viz].derived === 'roughness'
        || ((viz === 'hillshade' || viz === 'swiss') && colorOn);
    $('palette-wrap').hidden = !ramped;
    for (const b of $('palette').children) {
        b.setAttribute('aria-checked', b.dataset.palette === palette);
        const measure = VIZ[viz].derived === 'roughness' ? 'roughness'
                      : VIZ[viz].derived === 'slope' ? 'slope' : 'elevation';
        const max = measure === 'elevation' ? 8850 : DERIVED[measure].max;
        b.firstElementChild.style.background = diagonal(gradient(stopsFor(b.dataset.palette, theme, measure), max));
    }
    const url = new URL(location.href);
    if (palette === 'imhof') url.searchParams.delete('palette'); else url.searchParams.set('palette', palette);
    history.replaceState(history.state, '', url);
}

showViz();

// Light: fixed from the north-west, or the real sun for a date and Nepal time (UTC+5:45) at the map centre.
// While the time slider moves only the GPU hillshade turns (instant); cast shadows are worked out when it is let go.

const NPT = 5.75;
// Year box and date slider: days from 1 January of the chosen year, starting on today (in Nepal time).
const nowNpt = new Date(Date.now() + NPT * 3600e3);
let jan1 = 0;
function setYear(year) {
    year = Math.max(1900, Math.min(2100, Math.round(year) || nowNpt.getUTCFullYear()));
    $('sun-year').value = year;
    jan1 = Date.UTC(year, 0, 1);
    const days = (Date.UTC(year + 1, 0, 1) - jan1) / 864e5;
    $('sun-day').max = days - 1;
    $('sun-day').value = Math.min(+$('sun-day').value, days - 1);  // 31 Dec stays 31 Dec after a leap year
}
setYear(nowNpt.getUTCFullYear());
$('sun-day').value = Math.floor((nowNpt - jan1) / 864e5);
const dayLabel = day => new Date(jan1 + day * 864e5).toLocaleDateString('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'});
const hhmm = h => `${String(Math.floor(h) % 24).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}`;
let shadowUrl = '', swissTiles = '';

function sunNow() {
    const day = +$('sun-day').value, hour = parseFloat($('sun-hour').value);
    const c = map.getCenter();
    return {day, hour, ...sunPosition(jan1 + day * 864e5 + (hour - NPT) * 3600e3, c.lat, c.lng)};
}

// Turn the hillshade to the light: instant, every slider step.
function lightShading() {
    const sun = light === 'sun' ? sunNow() : null;
    if (sun) {
        $('sun-time').textContent = hhmm(sun.hour);
        $('sun-date-val').textContent = dayLabel(sun.day);
        $('sun-info').textContent = sun.altitude <= 0 ? 'The sun is below the horizon.'
            : `Sun ${sun.altitude.toFixed(0)}° above the horizon, from the ${compass(sun.azimuth)} (${Math.round(sun.azimuth)}°).`;
    }
    const day = sun && sun.altitude > 0;
    map.setPaintProperty('hillshade', 'hillshade-method', day ? 'basic' : 'standard');
    map.setPaintProperty('hillshade', 'hillshade-illumination-direction', day ? sun.azimuth : fixedAz);
    // Capped, for the reason in dem.js: a high sun leaves the relief no contrast and noon washes out. The
    // cast shadows still use the true height, because shadows really do shorten towards midday.
    map.setPaintProperty('hillshade', 'hillshade-illumination-altitude',
                         day ? Math.min(SHADE_ALT[1], Math.max(SHADE_ALT[0], sun.altitude)) : 45);
    map.setLayoutProperty('night', 'visibility', sun && !day ? 'visible' : 'none');
    return sun;
}

// The tiles that follow the sun: the cast shadows, and Swiss relief, which does its own shading.
function castShadowTiles() {
    const sun = light === 'sun' ? sunNow() : null;
    const on = !!sun && sun.altitude > 0;
    if (on) {
        // Rounded so small changes reuse tiles already made.
        const url = `derive://shadow/{z}/{x}/{y}?az=${Math.round(sun.azimuth)}&alt=${Math.round(sun.altitude * 2) / 2}`;
        if (url !== shadowUrl) {
            shadowUrl = url;
            map.getSource('derive-shadow').setTiles([url]);
        }
    }
    map.setLayoutProperty('shadow', 'visibility', on && viz !== 'swiss' ? 'visible' : 'none');
    const swissUrl = `derive://swiss/{z}/{x}/{y}?az=${on ? Math.round(sun.azimuth) : fixedAz}&alt=${on ? Math.round(sun.altitude) : 45}`;
    if (swissUrl !== swissTiles) {
        swissTiles = swissUrl;
        map.getSource('derive-swiss').setTiles([swissUrl]);
    }
    redrawDrape(map);
}

// After the slider is let go, the light changes or the map stops moving.
function castShadows() {
    lightShading();
    castShadowTiles();
}

function setLight(next) {
    light = next;
    $('light-nw').setAttribute('aria-pressed', light === 'nw');
    $('light-sun').setAttribute('aria-pressed', light === 'sun');
    $('sun-controls').hidden = light !== 'sun';
    for (const b of $('sun-popup').querySelectorAll('[data-az]')) {
        b.setAttribute('aria-pressed', light === 'nw' && +b.dataset.az === fixedAz);
    }
    setViz(viz);
    castShadows();
}

// A fixed direction to light from, the way a printed map is lit.
for (const b of document.querySelectorAll('[data-az]')) {
    b.onclick = () => {
        stopPlay();
        fixedAz = +b.dataset.az;
        setLight('nw');
    };
}

// Times of day, for the real sun.
for (const b of document.querySelectorAll('[data-hour]')) {
    b.onclick = () => {
        stopPlay();
        if (light !== 'sun') setLight('sun');
        $('sun-hour').value = b.dataset.hour;
        castShadows();
    };
}

$('light-nw').onclick = () => { stopPlay(); setLight('nw'); };
$('light-sun').onclick = () => setLight('sun');
$('sun-year').onchange = e => { stopPlay(); setYear(+e.target.value); castShadows(); };
for (const id of ['sun-hour', 'sun-day']) {
    $(id).oninput = () => {
        stopPlay();
        map.setLayoutProperty('shadow', 'visibility', 'none');  // the old shadows are wrong while the sun moves
        lightShading();
    };
    $(id).onchange = castShadows;
}

// Playing a day or a year. The hillshade follows the sun on the graphics card, which costs nothing, so the
// slider can run at the screen's own rate. The cast shadows and Swiss relief are tiles, so they are rebuilt a
// few times a second while playing and once more when it stops, which is what dragging the slider already did.
const PLAY = {
    hour: {slider: 'sun-hour', per: 4},   // hours of sun per second of watching: a day in 6 s
    day: {slider: 'sun-day', per: 12},    // days per second: a year in about 30 s
};
let playing = null, playFrom = 0, playAt = 0, playTiles = 0;

function stopPlay() {
    if (!playing) return;
    const was = playing;
    playing = null;
    $(`play-${was}`).setAttribute('aria-pressed', false);
    castShadows();
}

function step(now) {
    if (!playing) return;
    const {slider, per} = PLAY[playing];
    const el = $(slider);
    const lo = +el.min, hi = +el.max;
    const value = lo + (playFrom - lo + per * (now - playAt) / 1000) % (hi - lo + Number(el.step));
    el.value = value;
    lightShading();
    if (now - playTiles > 400) {   // the heavier tiles, a few times a second
        playTiles = now;
        castShadowTiles();
    }
    requestAnimationFrame(step);
}

for (const [name, {slider}] of Object.entries(PLAY)) {
    $(`play-${name}`).onclick = () => {
        const again = playing === name;
        stopPlay();
        if (again) return;
        if (light !== 'sun') setLight('sun');
        playing = name;
        playFrom = +$(slider).value;
        playAt = playTiles = performance.now();
        $(`play-${name}`).setAttribute('aria-pressed', true);
        requestAnimationFrame(step);
    };
}
map.on('moveend', () => { if (light === 'sun') castShadows(); });

// Peaks between a minimum and a maximum height.

let peaks = null;
const fmtH = v => `${Number(v).toLocaleString('en-US')} m`;

function updatePeaks() {
    let lo = +$('peak-min').value, hi = +$('peak-max').value;
    if (lo > hi) [lo, hi] = [hi, lo];
    $('peak-min-val').textContent = fmtH(lo);
    $('peak-max-val').textContent = fmtH(hi);
    const filter = ['all', ['>=', HEIGHT, lo], ['<=', HEIGHT, hi]];
    map.setFilter('peaks-dot', filter);
    map.setFilter('peaks-label', filter);
    if (peaks) {
        const n = peaks.features.filter(f => (f.properties.osm ?? f.properties.h) >= lo && (f.properties.osm ?? f.properties.h) <= hi).length;
        $('peak-count').textContent = `${n.toLocaleString('en-US')} summits in this range. Found in NepalDTM: each stands at least ` +
            '100 m above the ground linking it to any higher peak. Bigger dots stand out more. Names and surveyed heights from ' +
            'OpenStreetMap; ≈ marks NepalDTM heights' +
            (peaks.check ? `, which read a median ${Math.abs(peaks.check.all.median)} m low on named summits.` : '.');
    }
}

$('peaks-on').onchange = async e => {
    const on = e.target.checked;
    $('peak-controls').hidden = !on;
    if (on && !peaks) {
        $('peak-count').textContent = 'Loading summits…';
        peaks = await (await fetch('/data/summits.geojson', {cache: 'no-cache'})).json();
        map.getSource('peaks').setData(peaks);
    }
    for (const id of ['peaks-dot', 'peaks-label']) map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
    updatePeaks();
};
$('peak-min').oninput = updatePeaks;
$('peak-max').oninput = updatePeaks;
map.once('style.load', updatePeaks);
map.on('mouseenter', 'peaks-dot', () => { if (!profiling) map.getCanvas().style.cursor = 'pointer'; });
map.on('mouseleave', 'peaks-dot', () => { if (!profiling) restCursor(); });

function flyTo(lng, lat, zoom, bearing, pitch) {
    stopSpin();
    if (pitch > 0 && !map.getTerrain()) setMode(true, false);
    map.flyTo({center: [lng, lat], zoom, bearing, pitch, duration: 3000, essential: true});
    showPopup(null);
}

function allOfNepal() {
    stopSpin();
    map.easeTo({...allOfNepalCamera(), pitch: map.getTerrain() ? 45 : 0, bearing: 0, duration: 2500});
    showPopup(null);
}

{
    const list = $('places');
    const whole = document.createElement('button');
    whole.className = 'place-all';
    whole.textContent = 'All of Nepal';
    whole.onclick = allOfNepal;
    list.append(whole);
    for (const [province, places] of Object.entries(PLACES)) {
        const head = document.createElement('p');
        head.className = 'place-head';
        head.textContent = province;
        list.append(head);
        const row = document.createElement('div');
        row.className = 'place-row';
        for (const [name, ...view] of places) {
            const b = document.createElement('button');
            b.textContent = name;
            b.onclick = () => flyTo(...view);
            row.append(b);
        }
        list.append(row);
    }
}

function toggleSheet(open) {
    $('sheet-toggle').setAttribute('aria-expanded', open);
    $('panel').classList.toggle('collapsed', !open);
}
$('sheet-toggle').onclick = () => toggleSheet($('panel').classList.contains('collapsed'));
if (matchMedia('(max-width: 760px)').matches) toggleSheet(false);

// Hover readout from the map tiles

const fmtLL = ({lng, lat}) => `${lat.toFixed(4)}°N ${lng.toFixed(4)}°E`;
let hoverSeq = 0;

map.on('mousemove', e => {
    const seq = ++hoverSeq;
    const ll = e.lngLat;
    if (!inNepal(nepal, ll.lng, ll.lat)) {
        $('readout').innerHTML = `<span class="ll">${fmtLL(ll)}</span><span class="muted">outside Nepal</span>`;
        return;
    }
    const kind = VIZ[viz].derived;
    Promise.all([
        elevationAt(ll.lng, ll.lat, Math.round(map.getZoom()) + 3),
        kind ? derivedAt(kind, ll.lng, ll.lat, Math.min(TILE_MAX_ZOOM, Math.round(map.getZoom()))) : null,
    ]).then(([h, d]) => {
        if (seq !== hoverSeq) return;
        let extra = '';
        if (d && kind === 'slope') extra = `<span>slope ${d.slope.toFixed(0)}°</span>`;
        if (d && kind === 'aspect') extra = `<span>${d.slope < DERIVED.aspect.flat ? 'flat' : `faces ${compass(d.aspect)} (${Math.round(d.aspect)}°)`}</span>`;
        if (d && kind === 'roughness') extra = `<span>roughness ${d.roughness.toFixed(1)} m</span>`;
        $('readout').innerHTML = `<span class="ll">${fmtLL(ll)}</span>` +
            (h == null ? '<span class="muted">loading…</span>' : `<b>${fmtM(h)}</b>${extra}`);
    });
});

// Point inspector: exact 8 m values from the Cloud Optimized GeoTIFFs

proj4.defs('EPSG:32645', '+proj=utm +zone=45 +datum=WGS84 +units=m +no_defs');
const TILE = 24000; // processing grid, metres on UTM 45N
const images = new Map();

function tileName(e, n) {
    const f = v => v < 0 ? `-${String(-v).padStart(3, '0')}` : String(v).padStart(4, '0');
    return `E${f(Math.floor(e / TILE) * TILE / 1000)}N${f(Math.floor(n / TILE) * TILE / 1000)}`;
}

function cog(file) {
    if (!images.has(file)) {
        images.set(file, GeoTIFF.fromUrl(`${COG_URL}/${file}`).then(t => t.getImage()).catch(() => null));
    }
    return images.get(file);
}

async function readPixel(file, e, n) {
    const img = await cog(file);
    if (!img) return null;
    const [ox, oy] = img.getOrigin();
    const [rx, ry] = img.getResolution();
    const px = Math.floor((e - ox) / rx), py = Math.floor((n - oy) / ry);
    if (px < 0 || py < 0 || px >= img.getWidth() || py >= img.getHeight()) return null;
    const bands = await img.readRasters({window: [px, py, px + 1, py + 1]});
    const nodata = img.getGDALNoData();
    return Array.from(bands, b => b[0]).map(v => (nodata != null && v === nodata) || Number.isNaN(v) ? null : v);
}

// Slope and aspect at full resolution: Horn's method on the 3 x 3 cells of the 8 m bare-earth tile.
async function readSlope(file, e, n) {
    const img = await cog(file);
    if (!img) return null;
    const [ox, oy] = img.getOrigin();
    const [rx, ry] = img.getResolution();
    const px = Math.floor((e - ox) / rx), py = Math.floor((n - oy) / ry);
    if (px < 1 || py < 1 || px >= img.getWidth() - 1 || py >= img.getHeight() - 1) return null;
    const [w] = await img.readRasters({window: [px - 1, py - 1, px + 2, py + 2]});
    const nodata = img.getGDALNoData();
    if (Array.from(w).some(v => Number.isNaN(v) || (nodata != null && v === nodata))) return null;
    return horn(Array.from(w), Math.abs(rx));
}

const marker = new maplibregl.Marker({color: '#dc143c'});
let inspectSeq = 0;

async function inspect(ll, peak) {
    const seq = ++inspectSeq;
    marker.setLngLat(ll).addTo(map);
    $('inspect').hidden = false;
    if (!peak && !inNepal(nepal, ll.lng, ll.lat)) {  // border summits are kept: the 8 m tiles run 1 km past the border
        $('inspect-body').innerHTML = `<p class="ll">${fmtLL(ll)}</p><p>This point is outside Nepal.</p>`;
        return;
    }
    const [e, n] = proj4('EPSG:4326', 'EPSG:32645', [ll.lng, ll.lat]);
    const name = tileName(e, n);
    $('inspect-body').innerHTML = `<p class="ll">${fmtLL(ll)}</p><p class="muted">Reading tile ${name}…</p>`;
    const [dtm, dem, aux, shape] = await Promise.all([...['dtm', 'dem', 'aux'].map(k => readPixel(`${name}_${k}.tif`, e, n)),
        readSlope(`${name}_dtm.tif`, e, n)]);
    if (seq !== inspectSeq) return;
    const ground = dtm?.[0], surface = dem?.[0];
    const source = aux?.[0], err = aux?.[1];
    const rows = [];
    if (peak) {
        rows.push(['Summit', `<b>${peak.name || 'Unnamed'}</b>`]);
        if (peak.osm != null) rows.push(['Surveyed height', `<b>${peak.osm.toLocaleString('en-US')} m</b>`]);
        rows.push(['NepalDTM summit', fmtM(peak.h)]);
        rows.push(['Prominence', `at least ${fmtM(peak.prom)}`]);
    }
    if (ground != null) rows.push(['Bare earth', `<b>${ground.toFixed(1)} m</b>`]);
    if (surface != null) rows.push(['Surface', `${surface.toFixed(1)} m`]);
    if (ground != null && surface != null) rows.push(['Trees and buildings', `${Math.max(0, surface - ground).toFixed(1)} m`]);
    if (shape) {
        rows.push(['Slope (8 m)', `${shape.slope.toFixed(1)}°`]);
        rows.push(['Aspect', shape.slope < DERIVED.aspect.flat ? 'flat' : `${compass(shape.aspect)} (${Math.round(shape.aspect)}°)`]);
    }
    if (err != null) rows.push(['Expected error', err >= 255 ? '≥ 25.5 m' : `±${(err / 10).toFixed(1)} m`]);
    if (source != null) rows.push(['Source', source === 1 ? '<span class="tag hma">HMA 8 m stereo</span>'
                                          : '<span class="tag fill">Copernicus 30 m fill</span>']);
    const links = ['dtm', 'dem', 'aux'].map(k => `<a href="${COG_URL}/${name}_${k}.tif">${k}</a>`).join(' · ');
    $('inspect-body').innerHTML = `<p class="ll">${fmtLL(ll)}</p>` +
        (rows.length ? `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`
                     : '<p>No value at this point.</p>') +
        (peak ? summitNote(peak) : '') +
        `<p class="files">Tile <code>${name}</code> · download ${links}</p>` +
        '<p class="fine">Heights above the EGM2008 geoid.</p>';
}

// How far to trust a summit's NepalDTM height, from the summit check in summits.geojson.
function summitNote(peak) {
    const c = peaks?.check;
    if (!c || !peak.src) return '';
    const [g, where] = peak.src === 1 ? [c.hma_8m, 'the 8 m stereo data'] : [c.copernicus_fill, 'the 30 m Copernicus fill'];
    return `<p class="summit-note${peak.src === 2 ? ' warn' : ''}">This summit's height comes from ${where}. ` +
        `Named summits there read a median ${Math.abs(g.median)} m below their surveyed heights (${g.n} compared), ` +
        `and ${Math.round(g.share_more_than_50m_low * 100)}% are more than 50 m low. Sharp tops are flattened in an elevation model.</p>`;
}

$('inspect-close').onclick = () => { $('inspect').hidden = true; marker.remove(); };

// Elevation profile between two clicked points

let profiling = false, profileStart = null;
const hoverMarker = new maplibregl.Marker({element: Object.assign(document.createElement('div'), {className: 'profile-dot'})});

function setProfileLine(coords) {
    map.getSource('profile').setData({type: 'FeatureCollection', features: coords.length > 1
        ? [{type: 'Feature', properties: {}, geometry: {type: 'LineString', coordinates: coords}}] : []});
}

function setProfiling(on) {
    profiling = on;
    profileStart = null;
    $('tool-profile').setAttribute('aria-pressed', on);
    restCursor();
    $('tool-hint').textContent = on ? 'Click the start point, then the end point. Esc cancels.'
                                    : 'Click anywhere in Nepal for the exact 8 m values at that spot.';
}

$('tool-profile').onclick = () => setProfiling(!profiling);
document.addEventListener('keydown', e => { if (e.key === 'Escape' && profiling) { setProfiling(false); setProfileLine([]); } });
$('profile-close').onclick = () => { $('profile').hidden = true; setProfileLine([]); hoverMarker.remove(); };

function haversine([lng1, lat1], [lng2, lat2]) {
    const r = Math.PI / 180, a = Math.sin((lat2 - lat1) * r / 2) ** 2 +
        Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin((lng2 - lng1) * r / 2) ** 2;
    return 2 * 6371008.8 * Math.asin(Math.sqrt(a));
}

map.on('mousemove', e => {
    if (profiling && profileStart) setProfileLine([profileStart, e.lngLat.toArray()]);
});

map.on('click', e => {
    if (!profiling) {
        const [peak] = map.getLayoutProperty('peaks-dot', 'visibility') === 'visible'
            ? map.queryRenderedFeatures([[e.point.x - 6, e.point.y - 6], [e.point.x + 6, e.point.y + 6]], {layers: ['peaks-dot']}) : [];
        if (peak) return inspect({lng: peak.geometry.coordinates[0], lat: peak.geometry.coordinates[1]}, peak.properties);
        return inspect(e.lngLat);
    }
    if (!profileStart) { profileStart = e.lngLat.toArray(); return; }
    const a = profileStart, b = e.lngLat.toArray();
    setProfiling(false);
    setProfileLine([a, b]);
    drawProfile(a, b);
});

async function drawProfile(a, b) {
    const length = haversine(a, b);
    const zoom = length < 15000 ? TILE_MAX_ZOOM : length < 60000 ? 12 : length < 200000 ? 11 : 10;
    const N = 300;
    const pts = Array.from({length: N + 1}, (_, i) => [a[0] + (b[0] - a[0]) * i / N, a[1] + (b[1] - a[1]) * i / N]);
    $('profile').hidden = false;
    $('profile-stats').textContent = 'Loading…';
    $('profile-chart').innerHTML = '';
    const hs = await Promise.all(pts.map(([lng, lat]) => inNepal(nepal, lng, lat) ? elevationAt(lng, lat, zoom) : null));
    const data = pts.map((p, i) => ({p, d: length * i / N, h: hs[i]}));
    const valid = data.filter(s => s.h != null);
    if (valid.length < 2) { $('profile-stats').textContent = 'The line needs to cross Nepal.'; return; }

    let gain = 0, loss = 0;
    for (let i = 1; i < valid.length; i++) {
        const dh = valid[i].h - valid[i - 1].h;
        if (dh > 0) gain += dh; else loss -= dh;
    }
    const min = Math.min(...valid.map(s => s.h)), max = Math.max(...valid.map(s => s.h));
    const km = d => `${(d / 1000).toFixed(d < 10000 ? 2 : 1)} km`;
    $('profile-stats').innerHTML = `<span>${km(length)}</span><span>min ${fmtM(min)}</span><span>max ${fmtM(max)}</span>` +
        `<span>↑ ${fmtM(gain)}</span><span>↓ ${fmtM(loss)}</span>`;
    renderChart(data, length, min, max, km);
}

function renderChart(data, length, min, max, km) {
    const box = $('profile-chart');
    const W = box.clientWidth, H = box.clientHeight, L = 52, R = 12, T = 10, B = 24;
    const span = Math.max(max - min, 20), lo = min - span * 0.08, hi = max + span * 0.08;
    const x = d => L + (W - L - R) * d / length;
    const y = h => T + (H - T - B) * (1 - (h - lo) / (hi - lo));
    let line = '', area = '', open = false;
    for (const s of data) {
        if (s.h == null) { if (open) area += `L${x(s.d - length / 300)},${H - B}Z`; open = false; continue; }
        line += `${open ? 'L' : 'M'}${x(s.d).toFixed(1)},${y(s.h).toFixed(1)}`;
        area += open ? `L${x(s.d).toFixed(1)},${y(s.h).toFixed(1)}` : `M${x(s.d).toFixed(1)},${H - B}L${x(s.d).toFixed(1)},${y(s.h).toFixed(1)}`;
        open = true;
    }
    if (open) area += `L${x(data.at(-1).d)},${H - B}Z`;
    const yt = [lo + (hi - lo) * 0.1, (lo + hi) / 2, hi - (hi - lo) * 0.1];
    box.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="Elevation profile">
        ${yt.map(h => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(h)}" y2="${y(h)}"/><text class="ax" x="${L - 6}" y="${y(h) + 4}" text-anchor="end">${fmtM(h)}</text>`).join('')}
        <text class="ax" x="${L}" y="${H - 6}">0</text><text class="ax" x="${W - R}" y="${H - 6}" text-anchor="end">${km(length)}</text>
        <path class="area" d="${area}"/><path class="line" d="${line}"/>
        <line class="cursor" id="pc-line" y1="${T}" y2="${H - B}" visibility="hidden"/>
        <text class="cursor-label" id="pc-label" y="${T + 12}" visibility="hidden"></text>
        <rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent" id="pc-hit"/>
    </svg>`;
    const hit = $('pc-hit');
    hit.onpointermove = ev => {
        const rect = box.getBoundingClientRect();
        const d = Math.max(0, Math.min(length, (ev.clientX - rect.left - L) / (W - L - R) * length));
        const s = data[Math.round(d / length * (data.length - 1))];
        $('pc-line').setAttribute('x1', x(s.d)); $('pc-line').setAttribute('x2', x(s.d));
        $('pc-line').setAttribute('visibility', 'visible');
        const label = $('pc-label');
        label.textContent = s.h == null ? 'outside Nepal' : `${fmtM(s.h)} · ${km(s.d)}`;
        label.setAttribute('x', x(s.d) + (x(s.d) > W / 2 ? -8 : 8));
        label.setAttribute('text-anchor', x(s.d) > W / 2 ? 'end' : 'start');
        label.setAttribute('visibility', 'visible');
        hoverMarker.setLngLat(s.p).addTo(map);
    };
    hit.onpointerleave = () => {
        $('pc-line').setAttribute('visibility', 'hidden');
        $('pc-label').setAttribute('visibility', 'hidden');
        hoverMarker.remove();
    };
}

// Model version. Switching reloads the page with ?v=: the tile caches, workers and terrain source are all built
// for one archive, and the URL already carries the view (hash) and the visualization, so nothing is lost.
const VERSION_NOTES = {
    1: ['v1: held-out ICESat-2 RMSE 6.9 m (Copernicus 8.7 m).',
        '37% of Nepal has true 8 m detail; the rest is filled from 30 m Copernicus data.'],
    2: ['v2: held-out ICESat-2 RMSE 5.3 m (Copernicus 8.7 m). Stereo strips are jointly adjusted to ICESat-2 photons.',
        '39% of Nepal has true 8 m detail; the rest is filled from 30 m Copernicus data.'],
    g: ['G 1 m (test): v2 everywhere, plus generated 1 m detail over a 5 × 5.6 km test box in the Churia hills '
        + '(Chitwan) when zoomed past 14. The 1 m detail is generated, not measured; it keeps v2\'s accuracy there '
        + '(held-out ICESat-2 11.67 m vs 11.63 m; v2 is Copernicus fill under forest in this box).',
        'Generated detail is shown only inside the Churia test box; elsewhere this is v2.'],
};
$('ver-note').textContent = VERSION_NOTES[VERSION][0];
$('ver-fine').textContent = 'Map tiles go to 8.4 m per pixel (zoom 14). Clicking reads the full-resolution 8 m files of '
    + `${VERSIONS[VERSION].label}. ${VERSION_NOTES[VERSION][1]}`;
for (const v of [1, 2, 'g']) {
    const b = $(`ver-${v}`);
    b.setAttribute('aria-pressed', String(v === VERSION));
    b.onclick = () => {
        if (v === VERSION) return;
        const url = new URL(location.href);
        if (v === 2) url.searchParams.delete('v'); else url.searchParams.set('v', String(v));
        if (v === 'g') url.hash = '15/27.485/84.475';  // straight to the Churia test box
        location.assign(url);
    };
}

// Data and Paper menus in the top bar: one open at a time, closed by a click elsewhere or Escape.
const menus = [...document.querySelectorAll('.vbar .dd')];
menus.forEach(d => d.addEventListener('toggle', () => {
    if (d.open) menus.forEach(o => { if (o !== d) o.open = false; });
}));
document.addEventListener('click', e => menus.forEach(d => { if (!d.contains(e.target)) d.open = false; }));
document.addEventListener('keydown', e => { if (e.key === 'Escape') menus.forEach(d => { d.open = false; }); });
