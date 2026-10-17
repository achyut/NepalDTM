// Swiss relief and cast shadows on the graphics card, inside the tile worker.
//
// The CPU versions in dem.js walk every pixel in JavaScript: 56-147 ms for a shadow tile, and before that a
// canvas readback to turn the WebP into numbers. Here the nine WebP tiles go straight to the card as textures,
// the shader reads the Terrarium heights out of them, and the result comes back as one image. No readback,
// and the per-pixel work happens thousands of pixels at a time.
//
// Everything else is unchanged: these are still ordinary raster tiles, so terrain draping, the colour layer
// underneath and the edge cache all work as before. dem.js stays as the fallback where WebGL2 is missing, and
// it still answers the readout under the cursor, which needs exact numbers rather than pixels.

import {SHADE_ALT, swissWeights} from './dem.js';

const N = 256;
const GRID = 3 * N; // the tile plus its eight neighbours

// The card draws vertex y = +1 at the top of the output, while texture row 0 is the tile's north edge, so uv.y
// is flipped: without this each tile comes out upside down and every tile edge shows as a horizontal line.
const VERTEX = `#version 300 es
in vec2 pos;
out vec2 uv;
void main() {
    uv = vec2(pos.x, 1.0 - pos.y);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}`;

// Terrarium: height = R * 256 + G + B / 256 - 32768, with the texture holding 0..1 bytes.
const DECODE = `
float heightAt(sampler2D grid, ivec2 p) {
    vec3 c = texelFetch(grid, clamp(p, ivec2(0), ivec2(${GRID} - 1)), 0).rgb * 255.0;
    return c.r * 256.0 + c.g + c.b / 256.0 - 32768.0;
}`;

// Weights and colours match SWISS in dem.js, so the two paths draw the same picture.
const SWISS = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 colour;
uniform sampler2D grid;
uniform float az;          // where the light comes from, degrees from north
uniform float alt;         // how high it stands, degrees
uniform float flatShade;   // what lambert gives on flat ground
uniform float px;          // pixel width in metres
uniform vec3 weights;      // how much of the shading the 1, 5 and 15 pixel gradients carry
${DECODE}

const float RAD = 3.14159265 / 180.0;
const float LOCAL = 18.0;  // pixels: the width of landform the light is steered by
const float TURN = 35.0;   // degrees: how far the light may turn from the chosen direction
const float FOLLOW = 0.6;  // how much of the way it turns

float lambert(float dzdx, float dzdn, vec3 sun) {
    vec3 n = normalize(vec3(-dzdx, -dzdn, 1.0));
    return max(0.0, dot(n, sun));
}

void main() {
    ivec2 p = ivec2(uv * float(${N})) + ivec2(${N});

    // Cartographers do not light a whole sheet from one direction: they turn the light landform by landform so
    // each major slope catches it. Here a broad slope steers the light towards itself, by at most TURN degrees
    // so the chosen direction still governs the scene. Jenny et al., Cartographic Relief Shading with Neural
    // Networks (2020), name this among the practices their network learned from hand-drawn Swiss relief.
    int wide = int(LOCAL);
    float bspan = 2.0 * LOCAL * px;
    float bx = (heightAt(grid, p + ivec2(wide, 0)) - heightAt(grid, p - ivec2(wide, 0))) / bspan;
    float bn = -(heightAt(grid, p + ivec2(0, wide)) - heightAt(grid, p - ivec2(0, wide))) / bspan;
    float localAz = az;
    if (length(vec2(bx, bn)) > 0.02) {              // flat ground has no direction worth following
        float facing = degrees(atan(-bx, -bn));     // the way the broad slope looks, from north
        float turn = mod(facing - az + 540.0, 360.0) - 180.0;
        localAz = az + clamp(turn, -TURN, TURN) * FOLLOW;
    }
    float a = localAz * RAD, e = alt * RAD;
    vec3 sun = vec3(sin(a) * cos(e), cos(a) * cos(e), sin(e));

    float w[3] = float[3](weights.x, weights.y, weights.z);
    const int steps[3] = int[3](1, 5, 15);
    float shade = 0.0;
    for (int k = 0; k < 3; k++) {
        int d = steps[k];
        float span = 2.0 * float(d) * px;
        float dzdx = (heightAt(grid, p + ivec2(d, 0)) - heightAt(grid, p - ivec2(d, 0))) / span;
        float dzdn = -(heightAt(grid, p + ivec2(0, d)) - heightAt(grid, p - ivec2(0, d))) / span;
        shade += w[k] * lambert(dzdx, dzdn, sun);
    }
    // Lambert leaves more room below flat ground than above it, so stretch each side to -1..+1.
    float lit = shade >= flatShade ? (shade - flatShade) / (1.0 - flatShade)
                                   : (shade - flatShade) / flatShade;

    int wide = steps[2];
    float h = heightAt(grid, p);
    float mean = (heightAt(grid, p - ivec2(wide, 0)) + heightAt(grid, p + ivec2(wide, 0))
                + heightAt(grid, p - ivec2(0, wide)) + heightAt(grid, p + ivec2(0, wide))) / 4.0;
    lit += 0.22 * clamp((h - mean) / (0.35 * float(wide) * px), -1.0, 1.0);
    lit = clamp(lit, -1.0, 1.0);

    // Aerial perspective: the lowest ground loses a little contrast and takes on haze.
    float t = clamp((h - 600.0) / 3600.0, 0.0, 1.0);
    float haze = 0.1 * (1.0 - t);
    lit *= 1.0 - 0.2 * (1.0 - t);

    vec3 tint = lit < 0.0 ? vec3(44.0, 56.0, 78.0) / 255.0 : vec3(255.0, 253.0, 243.0) / 255.0;
    float strength = lit < 0.0 ? 0.68 * pow(-lit, 0.85) : 0.5 * pow(lit, 0.85);
    float total = strength + haze;
    if (total <= 0.0) { colour = vec4(0.0); return; }
    vec3 mixed = (tint * strength + vec3(186.0, 201.0, 217.0) / 255.0 * haze) / total;
    float a = min(0.8, total);
    colour = vec4(mixed * a, a);   // premultiplied, as the canvas expects
}`;

// Ground hidden from the sun by a ridge. Near pass at the tile's own resolution for crisp nearby shadows,
// far pass two zooms coarser so a ridge tens of kilometres away can still cast.
const SHADOW = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 colour;
uniform sampler2D grid;      // this zoom
uniform sampler2D coarse;    // two zooms out
uniform vec2 toSun;          // unit step towards the sun in grid pixels (y runs south)
uniform float fineRise;      // metres the ray climbs per fine pixel
uniform float coarseRise;
uniform vec2 coarseOrigin;   // this tile's corner inside the coarse grid, in coarse pixels
uniform float k;             // fine pixels per coarse pixel
${DECODE}

// March towards the sun; 0 lit, 1 in shadow, between near an edge.
float march(sampler2D tex, vec2 start, float rise, int steps, float soft) {
    float h0 = heightAt(tex, ivec2(start)) + 1.0;
    float shade = 0.0;
    for (int t = 1; t <= steps; t++) {
        vec2 q = start + toSun * float(t);
        if (q.x < 0.0 || q.y < 0.0 || q.x >= float(${GRID}) || q.y >= float(${GRID})) break;
        float above = heightAt(tex, ivec2(q)) - (h0 + float(t) * rise);
        if (above > 0.0) {
            shade = max(shade, above / soft);
            if (shade >= 1.0) return 1.0;
        }
    }
    return shade;
}

void main() {
    vec2 p = uv * float(${N}) + float(${N});
    float s = march(grid, p, fineRise, 64, max(3.0, 2.0 * fineRise));
    if (s < 1.0) {
        vec2 c = coarseOrigin + (uv * float(${N}) + 0.5) / k - 0.5 + float(${N});
        s = max(s, min(1.0, march(coarse, c, coarseRise, ${GRID}, max(4.0, 2.0 * coarseRise))));
    }
    float a = 0.5 * s;
    colour = vec4(vec3(16.0, 26.0, 54.0) / 255.0 * a, a);   // premultiplied
}`;

function compile(gl, source, type) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error(`shader: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
}

function link(gl, fragment) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, VERTEX, gl.VERTEX_SHADER));
    gl.attachShader(p, compile(gl, fragment, gl.FRAGMENT_SHADER));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        throw new Error(`program: ${gl.getProgramInfoLog(p)}`);
    }
    return p;
}

let gpu;

// null when WebGL2 is unavailable, so the caller falls back to the CPU path in dem.js.
export function initGpu() {
    if (gpu !== undefined) return gpu;
    gpu = null;
    try {
        const canvas = new OffscreenCanvas(N, N);
        const gl = canvas.getContext('webgl2', {premultipliedAlpha: true, antialias: false, depth: false});
        if (!gl) return gpu;
        const quad = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quad);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
        const programs = {swiss: link(gl, SWISS), shadow: link(gl, SHADOW)};
        for (const p of Object.values(programs)) {
            const loc = gl.getAttribLocation(p, 'pos');
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        }
        // Two grids: this zoom and, for shadows, two zooms out.
        const textures = [0, 1].map(unit => {
            const t = gl.createTexture();
            gl.activeTexture(gl.TEXTURE0 + unit);
            gl.bindTexture(gl.TEXTURE_2D, t);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GRID, GRID, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
                                  [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) {
                gl.texParameteri(gl.TEXTURE_2D, k, v);
            }
            return t;
        });
        gpu = {canvas, gl, programs, textures};
    } catch (err) {
        console.warn('GPU shading unavailable, falling back to the CPU path:', String(err));
        gpu = null;
    }
    return gpu;
}

// Blit the tile and its eight neighbours into one grid texture. A missing neighbour is left as it was, which
// reads as a flat edge; the centre tile is what the result is cut from, so an edge only affects the border.
function upload(unit, tiles) {
    const {gl, textures} = gpu;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, textures[unit]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    for (let i = 0; i < 9; i++) {
        const bmp = tiles[i];
        if (!bmp) continue;
        gl.texSubImage2D(gl.TEXTURE_2D, 0, (i % 3) * N, Math.floor(i / 3) * N, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    }
}

function draw(program, uniforms) {
    const {gl, canvas} = gpu;
    gl.useProgram(program);
    for (const [name, set] of Object.entries(uniforms)) set(gl.getUniformLocation(program, name));
    gl.viewport(0, 0, N, N);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return canvas.transferToImageBitmap();
}

export function swissOnGpu(tiles, az, alt, px, zoom) {
    const {gl, programs} = gpu;
    upload(0, tiles);
    // The same cartographic sun height as the CPU path: a near-overhead sun leaves the relief no contrast.
    const useAlt = Math.min(SHADE_ALT[1], Math.max(SHADE_ALT[0], alt));
    const w = swissWeights(zoom);
    return draw(programs.swiss, {
        grid: l => gl.uniform1i(l, 0),
        az: l => gl.uniform1f(l, az),
        alt: l => gl.uniform1f(l, useAlt),
        flatShade: l => gl.uniform1f(l, Math.sin(useAlt * Math.PI / 180)),
        px: l => gl.uniform1f(l, px),
        weights: l => gl.uniform3f(l, w[0], w[1], w[2]),
    });
}

export function shadowOnGpu(fine, coarse, az, alt, finePx, coarsePx, originX, originY, k) {
    const {gl, programs} = gpu;
    upload(0, fine);
    upload(1, coarse);
    const tan = Math.tan(alt * Math.PI / 180);
    return draw(programs.shadow, {
        grid: l => gl.uniform1i(l, 0),
        coarse: l => gl.uniform1i(l, 1),
        toSun: l => gl.uniform2f(l, Math.sin(az * Math.PI / 180), -Math.cos(az * Math.PI / 180)),
        fineRise: l => gl.uniform1f(l, tan * finePx),
        coarseRise: l => gl.uniform1f(l, tan * coarsePx),
        coarseOrigin: l => gl.uniform2f(l, originX, originY),
        k: l => gl.uniform1f(l, k),
    });
}
