// Serves the terrain tiles at /tiles/{z}/{x}/{y}.webp, and everything else from the built site.
//
// The map used to read byte ranges out of the 3 GB PMTiles archive directly. Cloudflare will not hold ranges
// of an object that size in its edge cache, so every tile travelled to R2 storage and came back in about
// 180 ms; a single view needs dozens of tiles. Served as its own small response, a tile is cacheable, and the
// second request for it is answered at the edge.
//
// The archive never changes once built, so tiles are immutable and can be cached for a year.

import { PMTiles } from "pmtiles";

const ARCHIVE = "NepalDTM.webp.pmtiles";
const TILE_PATH = /^\/tiles\/(\d+)\/(\d+)\/(\d+)\.webp$/;
const IMMUTABLE = "public, max-age=31536000, immutable";

// One R2 object, read in ranges. The archive's own directories are small and the PMTiles client keeps them,
// so most tile lookups cost a single range read.
class R2Source {
  constructor(bucket, key) {
    this.bucket = bucket;
    this.key = key;
  }

  getKey() {
    return this.key;
  }

  async getBytes(offset, length) {
    const object = await this.bucket.get(this.key, { range: { offset, length } });
    if (!object) throw new Error(`${this.key}: no object`);
    return { data: await object.arrayBuffer() };
  }
}

// The archive is immutable, so its directory cache is shared across requests in this isolate rather than
// rebuilt each time. It holds no request state.
let archive;

function open(env) {
  archive ??= new PMTiles(new R2Source(env.TILES, ARCHIVE));
  return archive;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const tile = TILE_PATH.exec(url.pathname);
    if (!tile) return env.ASSETS.fetch(request);
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405, headers: { allow: "GET, HEAD" } });
    }

    const cache = caches.default;
    const hit = await cache.match(request);
    if (hit) return hit;

    const [, z, x, y] = tile;
    let response;
    try {
      const result = await open(env).getZxy(Number(z), Number(x), Number(y));
      // Tiles are left out where there is no terrain. A 404 says so and, unlike a 204, the cache will hold
      // it, so the empty ground around Nepal costs one lookup rather than one per visit.
      response = result
        ? new Response(result.data, { headers: { "content-type": "image/webp", "cache-control": IMMUTABLE } })
        : new Response("", { status: 404, headers: { "cache-control": IMMUTABLE } });
    } catch (err) {
      console.error({ message: "tile read failed", tile: `${z}/${x}/${y}`, error: String(err) });
      return new Response("Tile unavailable", { status: 502, headers: { "cache-control": "no-store" } });
    }

    ctx.waitUntil(cache.put(request, response.clone()));
    return response;
  },
};
