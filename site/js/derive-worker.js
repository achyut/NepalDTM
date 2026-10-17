// Tile worker: computes slope, aspect, roughness and shadow tiles off the main thread, so the map stays smooth.
// Each worker reads the terrain archive itself and keeps its own cache of decoded heights.

import {TILE_SIZE, heightSource, renderDerived, renderShadow, renderSwiss} from './dem.js';

let tileHeights;

self.onmessage = async ({data: m}) => {
    if (m.type === 'init') {
        // The worker fetches the same cached tile URLs as the page, so the browser cache is shared.
        tileHeights = heightSource((z, x, y) => `${m.origin}/tiles/${z}/${x}/${y}.webp`, 500);
        return;
    }
    try {
        const render = {shadow: renderShadow, swiss: renderSwiss}[m.kind];
        const rgba = render ? await render(tileHeights, m.z, m.x, m.y, m.az, m.alt)
                            : await renderDerived(tileHeights, m.kind, m.z, m.x, m.y, m.theme);
        if (!rgba) return self.postMessage({id: m.id});
        const bmp = await createImageBitmap(new ImageData(rgba, TILE_SIZE, TILE_SIZE));
        self.postMessage({id: m.id, bmp}, [bmp]);
    } catch (e) {
        self.postMessage({id: m.id, error: String(e)});
    }
};
