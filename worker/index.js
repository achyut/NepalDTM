// Serves the terrain tiles at /tiles/{z}/{x}/{y}.webp (v1) and /tiles/v2/{z}/{x}/{y}.webp (v2), and everything
// else from the built site.
//
// The map used to read byte ranges out of the 3 GB PMTiles archive directly. Cloudflare will not hold ranges
// of an object that size in its edge cache, so every tile travelled to R2 storage and came back in about
// 180 ms; a single view needs dozens of tiles. Served as its own small response, a tile is cacheable, and the
// second request for it is answered at the edge.
//
// The archive never changes once built, so tiles are immutable and can be cached for a year.

import { PMTiles } from "pmtiles";

// One archive per model version, both in the same bucket.
const ARCHIVES = { 1: "NepalDTM.webp.pmtiles", 2: "v2/NepalDTM_v2.webp.pmtiles", g: "g/NepalDTM_G_test.webp.pmtiles" };
const TILE_PATH = /^\/tiles\/(?:(v2|g)\/)?(\d+)\/(\d+)\/(\d+)\.webp$/;
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

// The archives are immutable, so their directory caches are shared across requests in this isolate rather than
// rebuilt each time. They hold no request state.
const archives = {};

function open(env, version) {
  archives[version] ??= new PMTiles(new R2Source(env.TILES, ARCHIVES[version]));
  return archives[version];
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

    const [, which, z, x, y] = tile;
    const version = which === "v2" ? 2 : which === "g" ? "g" : 1;
    let response;
    try {
      const result = await open(env, version).getZxy(Number(z), Number(x), Number(y));
      // Tiles are left out where there is no terrain. A 404 says so and, unlike a 204, the cache will hold
      // it, so the empty ground around Nepal costs one lookup rather than one per visit.
      response = result
        ? new Response(result.data, { headers: { "content-type": "image/webp", "cache-control": IMMUTABLE } })
        : new Response("", { status: 404, headers: { "cache-control": IMMUTABLE } });
    } catch (err) {
      console.error({ message: "tile read failed", tile: `${which ? which + "/" : ""}${z}/${x}/${y}`, error: String(err) });
      return new Response("Tile unavailable", { status: 502, headers: { "cache-control": "no-store" } });
    }

    ctx.waitUntil(cache.put(request, response.clone()));
    return response;
  },
};
