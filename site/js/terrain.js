// Shared map setup for the home page and the viewer: the NepalDTM tiles, the colour ramp,
// the mask that hides everything outside Nepal, and point elevation read from the tiles.
// Expects the maplibregl global from its script tag.

import {TILE_MAX_ZOOM, TILE_SIZE, heightSource, locate} from './dem.js';
import {stopsFor} from './palettes.js';

export {TILE_MAX_ZOOM}; // highest zoom in the archive, about 8.4 m per pixel over Nepal

export const DATA_URL = 'https://nepaldtm-data.achyutpdl.com';
// Tiles come from this site's own worker, which reads the archive in R2 and lets Cloudflare cache each tile
// at the edge. Reading byte ranges from the 3 GB archive missed that cache on every single tile.
export const TILE_URL = '/tiles/{z}/{x}/{y}.webp';
export const tileUrl = (z, x, y) => `/tiles/${z}/${x}/${y}.webp`;
export const NEPAL_BOUNDS = [[80.06, 26.35], [88.2, 30.45]];
export const MAX_BOUNDS = [[77.0, 24.0], [91.3, 32.6]];

// Tiles are left out where there is no terrain (the worker answers 404), and MapLibre never finishes a
// terrain tile that comes back empty, so those get a flat 0 m tile: the value the archive uses outside the
// 24 km grid. Terrarium 0 m is rgb(128, 0, 0).
let flatTile;
function flat() {
    flatTile ??= (async () => {
        const c = new OffscreenCanvas(TILE_SIZE, TILE_SIZE);
        const ctx = c.getContext('2d');
        ctx.fillStyle = 'rgb(128, 0, 0)';
        ctx.fillRect(0, 0, TILE_SIZE, TILE_SIZE);
        return (await c.convertToBlob({type: 'image/png'})).arrayBuffer();
    })();
    return flatTile;
}

maplibregl.addProtocol('terrain', async (params, abort) => {
    const r = await fetch(params.url.replace(/^terrain:\/\//, ''), {signal: abort.signal});
    if (r.status === 404) return {data: await flat()};
    if (!r.ok) throw new Error(`tile ${r.status}`);
    return {data: await r.arrayBuffer()};
});

// The height ramp of the chosen scheme. palettes.js holds them all; imhof is the cartographic default.
export const reliefStops = (palette, theme) => stopsFor(palette, theme, 'elevation');

// Surroundings: the page around the map, the colour hiding everything outside Nepal, and the border.
export const THEMES = {
    light: {outside: '#e3e9f2', border: '#003893', horizon: '#f4f7fb'},
    dark: {outside: '#0a1528', border: '#b4bfd2', horizon: '#132442'},
    blue: {outside: '#003893', border: '#b4bfd2', horizon: '#0a429d'},
};

const sky = ({outside, horizon}) => ({'sky-color': outside, 'horizon-color': horizon, 'fog-color': outside,
    'sky-horizon-blend': 0.6, 'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.7, 'atmosphere-blend': 0});

// In 3D the draped layers are cached as textures; re-applying the terrain after the next frame redraws them.
export function redrawDrape(map) {
    const terrain = map.getTerrain();
    if (terrain) {
        map.once('render', () => {
            map.setTerrain(null);
            map.setTerrain(terrain);
        });
    }
    map.triggerRepaint();
}

export function applyTheme(map, theme) {
    map.setPaintProperty('background', 'background-color', theme.outside);
    map.setPaintProperty('mask', 'fill-color', theme.outside);
    map.setPaintProperty('border', 'line-color', theme.border);
    map.setSky(sky(theme));
    redrawDrape(map);
}

export async function loadNepal() {
    const r = await fetch('/data/nepal.geojson');
    if (!r.ok) throw new Error(`boundary: HTTP ${r.status}`);
    return r.json();
}

function polygons(fc) {
    return fc.features.flatMap(f => f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates);
}

function signedArea(ring) {
    let a = 0;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) a += (ring[j][0] - ring[i][0]) * (ring[j][1] + ring[i][1]);
    return a;
}

// A large box with Nepal cut out. MapLibre reads a ring as a hole only when it winds against the outer ring.
export function outsideMask(fc) {
    const outer = [[60, 10], [110, 10], [110, 45], [60, 45], [60, 10]];
    const holes = polygons(fc).map(p => {
        const ring = p[0].slice();
        return Math.sign(signedArea(ring)) === Math.sign(signedArea(outer)) ? ring.reverse() : ring;
    });
    return {type: 'Feature', properties: {}, geometry: {type: 'Polygon', coordinates: [outer, ...holes]}};
}

export function inNepal(fc, lng, lat) {
    return polygons(fc).some(([ring]) => {
        let inside = false;
        for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const [xi, yi] = ring[i], [xj, yj] = ring[j];
            if ((yi > lat) !== (yj > lat) && lng < (xj - xi) * (lat - yi) / (yj - yi) + xi) inside = !inside;
        }
        return inside;
    });
}

export function terrainStyle(nepal, {exaggeration = 1.5, theme = THEMES.light, palette = 'imhof', mode = 'light'} = {}) {
    const {outside, border} = theme;
    const dem = {type: 'raster-dem', tiles: [`terrain://${TILE_URL}`], tileSize: TILE_SIZE, encoding: 'terrarium',
                 minzoom: 5, maxzoom: TILE_MAX_ZOOM, bounds: [79.79, 26.08, 88.3, 30.81]};
    return {
        version: 8,
        transition: {duration: 0, delay: 0},
        sources: {
            terrain: {...dem},
            relief: {...dem},
            nepal: {type: 'geojson', data: nepal},
            mask: {type: 'geojson', data: outsideMask(nepal)},
        },
        layers: [
            {id: 'background', type: 'background', paint: {'background-color': outside}},
            {id: 'land', type: 'fill', source: 'nepal', paint: {'fill-color': '#eef0f2'}},
            {id: 'relief', type: 'color-relief', source: 'relief',
             paint: {'color-relief-color': ['interpolate', ['linear'], ['elevation'], ...reliefStops(palette, mode).flat()]}},
            {id: 'hillshade', type: 'hillshade', source: 'relief',
             paint: {'hillshade-exaggeration': ['interpolate', ['linear'], ['zoom'], 5, 0.7, 9, 0.6, 12, 0.5, 15, 0.35],
                     'hillshade-shadow-color': 'rgb(64, 82, 104)', 'hillshade-highlight-color': 'rgb(255, 249, 214)',
                     'hillshade-accent-color': 'rgb(88, 102, 118)', 'hillshade-illumination-direction': 315,
                     'hillshade-illumination-anchor': 'map'}},
            {id: 'mask', type: 'fill', source: 'mask', paint: {'fill-color': outside}},
            {id: 'border', type: 'line', source: 'nepal',
             paint: {'line-color': border, 'line-width': ['interpolate', ['linear'], ['zoom'], 6, 1.2, 12, 2.2]}},
        ],
        terrain: {source: 'terrain', exaggeration},
        sky: sky(theme),
    };
}

// Decoded heights, shared by the hover readout, the profile tool and slope/aspect/roughness under the cursor.
export const tileHeights = heightSource(tileUrl);

export async function elevationAt(lng, lat, zoom = TILE_MAX_ZOOM) {
    const {z, x, y, px, py} = locate(lng, lat, zoom);
    const h = await tileHeights(z, x, y);
    return h ? h[py * TILE_SIZE + px] : null;
}

export const fmtM = v => `${Math.round(v).toLocaleString('en-US')} m`;
