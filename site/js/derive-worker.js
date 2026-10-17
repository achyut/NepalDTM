// Tile worker: computes slope, aspect, roughness and shadow tiles off the main thread, so the map stays smooth.
// Each worker reads the terrain archive itself and keeps its own cache of decoded heights.

import {PMTiles} from 'https://cdn.jsdelivr.net/npm/pmtiles@4.4.0/+esm';
import {TILE_SIZE, heightSource, renderDerived, renderShadow} from './dem.js';

let tileHeights;

self.onmessage = async ({data: m}) => {
    if (m.type === 'init') {
        tileHeights = heightSource(new PMTiles(m.url), 500);
        return;
    }
    try {
        const rgba = m.kind === 'shadow'
            ? await renderShadow(tileHeights, m.z, m.x, m.y, m.az, m.alt)
            : await renderDerived(tileHeights, m.kind, m.z, m.x, m.y, m.theme);
        if (!rgba) return self.postMessage({id: m.id});
        const bmp = await createImageBitmap(new ImageData(rgba, TILE_SIZE, TILE_SIZE));
        self.postMessage({id: m.id, bmp}, [bmp]);
    } catch (e) {
        self.postMessage({id: m.id, error: String(e)});
    }
};
