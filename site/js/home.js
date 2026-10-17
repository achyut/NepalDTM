import {NEPAL_BOUNDS, THEMES, loadNepal, terrainStyle} from './terrain.js';
import {initThemeToggle} from './theme.js';

initThemeToggle();

for (const b of document.querySelectorAll('[data-copy]')) {
    b.onclick = async () => {
        await navigator.clipboard.writeText(b.nextElementSibling.textContent);
        b.textContent = 'Copied';
        setTimeout(() => { b.textContent = 'Copy'; }, 1500);
    };
}

// Hero: the whole country in 3D, turning slowly.
const nepal = await loadNepal();
const map = new maplibregl.Map({
    container: 'hero-map',
    style: terrainStyle(nepal, {exaggeration: 2, theme: THEMES.blue}),
    bounds: NEPAL_BOUNDS,
    // Wide screens: the country sits to the right of the headline. Phones: it gets its own strip above it.
    fitBoundsOptions: {padding: innerWidth > 900
        ? {top: 90, bottom: 90, left: Math.round(innerWidth * 0.48), right: 32}
        : {top: 16, bottom: 16, left: 12, right: 12}},
    interactive: false,
    attributionControl: false,
});
map.once('load', () => {
    map.easeTo({pitch: 50, duration: 0});
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const spin = () => map.easeTo({bearing: map.getBearing() + 20, duration: 40000, easing: t => t});
    map.on('moveend', spin);
    spin();
});
