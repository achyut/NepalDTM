// Slope, aspect, roughness and shadow tiles for MapLibre through a derive:// protocol
// (derive://slope/{z}/{x}/{y}?t=light, derive://shadow/{z}/{x}/{y}?az=..&alt=..). The work runs in a pool of
// workers (derive-worker.js), so panning and dragging stay smooth. Only tiles MapLibre asks for are made,
// the newest request first, and a tile scrolled away before its turn is dropped. Tiles that share a coarser
// parent go to the same worker, so its cache of decoded heights is reused.

import {DERIVED, TILE_SIZE, aspectColor, compass, derivedAt as derivedAtWith, horn, sunPosition} from './dem.js';
import {PMTILES_URL, tileHeights} from './terrain.js';

export {DERIVED, aspectColor, compass, horn, sunPosition};

const SIZE = Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 1));
const queue = [];
const running = new Map();
let nextId = 1;

const workers = Array.from({length: SIZE}, () => {
    const w = new Worker(new URL('./derive-worker.js', import.meta.url), {type: 'module'});
    w.postMessage({type: 'init', url: PMTILES_URL});
    w.job = null;
    w.lastKey = '';
    w.onmessage = ({data}) => {
        const job = running.get(data.id);
        running.delete(data.id);
        w.job = null;
        if (job) {
            if (data.error) job.reject(new Error(data.error));
            else job.resolve(data.bmp || null);
        }
        pump();
    };
    return w;
});

function pump() {
    for (const w of workers) {
        if (w.job || !queue.length) continue;
        // Prefer the newest job near the worker's last one; otherwise the newest job.
        let at = queue.length - 1;
        for (let i = queue.length - 1; i >= 0; i--) if (queue[i].key === w.lastKey) { at = i; break; }
        const [job] = queue.splice(at, 1);
        w.job = job;
        w.lastKey = job.key;
        running.set(job.msg.id, job);
        w.postMessage(job.msg);
    }
}

let blank;
const empty = () => blank ??= createImageBitmap(new ImageData(TILE_SIZE, TILE_SIZE));

maplibregl.addProtocol('derive', (params, abort) => new Promise((resolve, reject) => {
    const [, kind, z, x, y, query] = params.url.match(/^derive:\/\/(\w+)\/(\d+)\/(\d+)\/(\d+)(?:\?(.*))?$/);
    const q = new URLSearchParams(query);
    const msg = {id: nextId++, kind, z: +z, x: +x, y: +y, theme: q.get('t') === 'dark' ? 'dark' : 'light',
                 az: +q.get('az'), alt: +q.get('alt')};
    const job = {msg, key: `${+z - 2}/${x >> 2}/${y >> 2}`,
                 resolve: async bmp => resolve({data: bmp || await empty()}), reject};
    queue.push(job);
    abort.signal.addEventListener('abort', () => {
        const i = queue.indexOf(job);
        if (i >= 0) queue.splice(i, 1);
        reject(new DOMException('Tile no longer needed', 'AbortError'));
    });
    pump();
}));

// Raster source for one measure.
export function derivedSource(kind, bounds, query = '') {
    return {type: 'raster', tiles: [`derive://${kind}/{z}/{x}/{y}${query}`], tileSize: TILE_SIZE, minzoom: 5, maxzoom: 14, bounds};
}

// The measure under a point (for the hover readout), computed on the page from the cached tiles.
export const derivedAt = (kind, lng, lat, zoom) => derivedAtWith(tileHeights, kind, lng, lat, zoom);
