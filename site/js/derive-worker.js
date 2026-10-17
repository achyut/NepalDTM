// Tile worker: makes the slope, aspect, roughness, Swiss relief and shadow tiles away from the main thread,
// so panning and dragging stay smooth.
//
// Swiss relief and the shadows run on the graphics card (gpu.js) where WebGL2 is available: the WebP tiles go
// straight to the card and the shader reads the heights out of them, which skips both the canvas readback and
// the per-pixel JavaScript. Everything falls back to the CPU versions in dem.js otherwise, and slope, aspect
// and roughness stay on the CPU because they are only a few milliseconds each.

import {TILE_SIZE, bitmapSource, heightSource, pixelSize, renderDerived, renderShadow, renderSwiss} from './dem.js';
import {initGpu, shadowOnGpu, swissOnGpu} from './gpu.js';

let tileHeights, tileBitmap, gpu;

// The tile and its eight neighbours, in reading order, as images for the card.
const neighbours = (z, x, y) =>
    Promise.all([-1, 0, 1].flatMap(dy => [-1, 0, 1].map(dx => tileBitmap(z, x + dx, y + dy))));

self.onmessage = async ({data: m}) => {
    if (m.type === 'init') {
        const url = (z, x, y) => `${m.origin}/tiles/${z}/${x}/${y}.webp`;
        tileHeights = heightSource(url, 500);
        tileBitmap = bitmapSource(url, 500);
        gpu = initGpu();
        return;
    }
    try {
        let bmp;
        if (gpu && (m.kind === 'swiss' || m.kind === 'shadow')) {
            bmp = await onGpu(m);
        } else {
            const render = {shadow: renderShadow, swiss: renderSwiss}[m.kind];
            const rgba = render ? await render(tileHeights, m.z, m.x, m.y, m.az, m.alt)
                                : await renderDerived(tileHeights, m.kind, m.z, m.x, m.y, m.theme, m.palette);
            if (rgba) bmp = await createImageBitmap(new ImageData(rgba, TILE_SIZE, TILE_SIZE));
        }
        if (!bmp) return self.postMessage({id: m.id});
        self.postMessage({id: m.id, bmp}, [bmp]);
    } catch (e) {
        self.postMessage({id: m.id, error: String(e)});
    }
};

async function onGpu(m) {
    const {kind, z, x, y, az, alt} = m;
    const fine = await neighbours(z, x, y);
    if (!fine[4]) return null;               // no centre tile means no terrain here
    if (kind === 'swiss') return swissOnGpu(fine, az, alt, pixelSize(z, y), z);
    if (alt <= 0) return null;               // the sun is down: nothing casts
    // Long shadows come from two zooms out, where a ridge tens of kilometres away still reaches.
    const zc = Math.max(5, z - 2), k = 2 ** (z - zc), cx = x >> (z - zc), cy = y >> (z - zc);
    const coarse = await neighbours(zc, cx, cy);
    const span = TILE_SIZE / k;
    return shadowOnGpu(fine, coarse, az, alt, pixelSize(z, y), pixelSize(zc, cy),
                       (x - cx * k) * span, (y - cy * k) * span, k);
}
