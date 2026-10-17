// Colour schemes for the layers that carry a ramp: height, slope and roughness.
//
// Each scheme gives a light and a dark version, because a ramp that starts near black vanishes on a dark page
// and one that starts near white vanishes on a light one. Values are stops along the measure, so the same
// scheme works whatever the range.
//
//   imhof     the cartographic default: Terai greens through hill ochres and rock browns to snow, after the
//             hypsometric tints of Swiss atlases
//   viridis   perceptually uniform, the matplotlib default (Smith and van der Walt); colour-blind safe, and
//             it still reads in black and white
//   magma     perceptually uniform and warm, from the same family
//   inferno   the same family again, pushed further towards fire
//   terrain   the familiar school-atlas green to brown to white
//   grey      no hue at all: the greyscale of a printed relief map, and what the others look like to a
//             monochrome printer
//
// Only imhof is tuned for Nepal's heights; the others are even ramps and will put a colour boundary wherever
// the numbers fall.

const ramp = (stops, colours) => stops.map((v, i) => [v, colours[i]]);

// Height, 0 to 8,850 m. Nepal spans the lot, so the stops are placed where the country's land actually is.
const H = [0, 200, 600, 1200, 2000, 3000, 4000, 5000, 6000, 8850];
const EVEN = [0, 983, 1967, 2950, 3933, 4917, 5900, 6883, 7867, 8850];

export const PALETTES = {
    imhof: {
        label: 'Imhof',
        note: 'Hypsometric tints after the Swiss atlases: Terai greens, hill ochres, rock browns, snow.',
        light: {
            elevation: ramp(H, ['#a7c896', '#bcd5a8', '#d3e0bb', '#e8e8c9', '#efe3c6',
                                '#e3d1b5', '#d6cbbf', '#e0dedb', '#f1f0ee', '#ffffff']),
            slope: [[0, '#f4f7fb'], [5, '#dfe8c8'], [12, '#f1e2a0'], [20, '#f2b86c'], [30, '#e07a4f'],
                    [40, '#c03a3e'], [50, '#7d1d3a'], [60, '#3d0f2a']],
            roughness: [[0, '#f4f7fb'], [2, '#d3dae6'], [5, '#a9bbdc'], [10, '#6f92cc'], [20, '#2f63b8'],
                        [30, '#003893'], [40, '#0a1528']],
        },
        dark: {
            elevation: ramp(H, ['#2c4a39', '#3a5c42', '#4d6b46', '#6b7a4a', '#897f52',
                                '#9a7f5e', '#a89080', '#bfc0bd', '#dcdedd', '#ffffff']),
            slope: [[0, '#16213a'], [8, '#3b1f5c'], [16, '#6b1f6f'], [24, '#a12f5e'], [32, '#d24c3a'],
                    [42, '#f07a1f'], [52, '#f9b425'], [60, '#fcf6a3']],
            roughness: [[0, '#16213a'], [2, '#1d3462'], [5, '#23508f'], [10, '#2f6fc0'], [20, '#5a9be0'],
                        [30, '#9cc4f2'], [40, '#e6f0ff']],
        },
    },
    viridis: {
        label: 'Viridis',
        note: 'Perceptually uniform and colour-blind safe: equal steps in the number look like equal steps in '
            + 'the colour. The matplotlib default.',
        light: {
            elevation: ramp(EVEN, ['#440154', '#482878', '#3e4989', '#31688e', '#26828e',
                                   '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#440154', '#46327e', '#365c8d', '#277f8e',
                                                         '#1fa187', '#4ac16d', '#a0da39', '#fde725']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#440154', '#46327e', '#365c8d', '#277f8e',
                                                             '#1fa187', '#4ac16d', '#a0da39', '#fde725']),
        },
        dark: {
            // Lifted off the darkest end, which would disappear against a dark page.
            elevation: ramp(EVEN, ['#2d4f6b', '#31688e', '#2a788e', '#26828e', '#1f9e89',
                                   '#35b779', '#6ece58', '#b5de2b', '#dcea4e', '#fde725']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#2d4f6b', '#31688e', '#277f8e', '#1fa187',
                                                         '#4ac16d', '#a0da39', '#dcea4e', '#fde725']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#2d4f6b', '#31688e', '#277f8e', '#1fa187',
                                                             '#4ac16d', '#a0da39', '#dcea4e', '#fde725']),
        },
    },
    magma: {
        label: 'Magma',
        note: 'Perceptually uniform and warm, from the same family as Viridis.',
        light: {
            elevation: ramp(EVEN, ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f',
                                   '#cd4071', '#f1605d', '#fd9668', '#feca8d', '#fcfdbf']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#000004', '#2c115f', '#721f81', '#b63679',
                                                         '#f1605d', '#feaf77', '#fed395', '#fcfdbf']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#000004', '#2c115f', '#721f81', '#b63679',
                                                             '#f1605d', '#feaf77', '#fed395', '#fcfdbf']),
        },
        dark: {
            elevation: ramp(EVEN, ['#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968',
                                   '#f1605d', '#fb8861', '#fd9f6c', '#feca8d', '#fcfdbf']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#3b0f70', '#721f81', '#b63679', '#de4968',
                                                         '#f1605d', '#feaf77', '#fed395', '#fcfdbf']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#3b0f70', '#721f81', '#b63679', '#de4968',
                                                             '#f1605d', '#feaf77', '#fed395', '#fcfdbf']),
        },
    },
    inferno: {
        label: 'Inferno',
        note: 'The same family again, pushed towards fire. Strong where you want the high values to shout.',
        light: {
            elevation: ramp(EVEN, ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60',
                                   '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#000004', '#320a5e', '#781c6d', '#bb3754',
                                                         '#ed6925', '#fbb61a', '#f6d746', '#fcffa4']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#000004', '#320a5e', '#781c6d', '#bb3754',
                                                             '#ed6925', '#fbb61a', '#f6d746', '#fcffa4']),
        },
        dark: {
            elevation: ramp(EVEN, ['#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925',
                                   '#fb9b06', '#f7d13d', '#f9e07f', '#fbf0a5', '#fcffa4']),
            slope: ramp([0, 9, 17, 26, 34, 43, 51, 60], ['#4a0c6b', '#781c6d', '#bb3754', '#ed6925',
                                                         '#fbb61a', '#f6d746', '#faeb8c', '#fcffa4']),
            roughness: ramp([0, 6, 11, 17, 23, 29, 34, 40], ['#4a0c6b', '#781c6d', '#bb3754', '#ed6925',
                                                             '#fbb61a', '#f6d746', '#faeb8c', '#fcffa4']),
        },
    },
    terrain: {
        label: 'Atlas',
        note: 'The familiar school-atlas scheme: green lowlands, brown uplands, white summits.',
        light: {
            elevation: ramp(H, ['#3a7d3a', '#5a9b4a', '#8bb85e', '#c3cd77', '#e3d08a',
                                '#c9a063', '#a9754b', '#8d6a56', '#c9c4bd', '#ffffff']),
            slope: [[0, '#eef6ea'], [8, '#c9e0b4'], [16, '#e8dd92'], [24, '#e0b063'], [32, '#cb7f4a'],
                    [42, '#a9503c'], [52, '#75302f'], [60, '#3f1a1c']],
            roughness: [[0, '#f2f6f2'], [4, '#cfe0cb'], [9, '#a8c3a2'], [15, '#7fa37c'], [22, '#55805a'],
                        [31, '#2e5a40'], [40, '#14331f']],
        },
        dark: {
            elevation: ramp(H, ['#1d4426', '#2b5c30', '#427036', '#6b8040', '#8c8550',
                                '#8d6a45', '#7c5d46', '#8a8178', '#c6c3bc', '#ffffff']),
            slope: [[0, '#16271c'], [8, '#27432a'], [16, '#4a6232'], [24, '#7c7a34'], [32, '#b0813a'],
                    [42, '#d8893f'], [52, '#efac5e'], [60, '#ffe0a6']],
            roughness: [[0, '#16271c'], [4, '#23402b'], [9, '#355c3a'], [15, '#4e7d4b'], [22, '#73a166'],
                        [31, '#a3c58d'], [40, '#dcebc8']],
        },
    },
    grey: {
        label: 'Greyscale',
        note: 'No hue at all, the way a relief map prints in black and white: value alone carries the number.',
        light: {
            elevation: ramp(H, ['#3f444c', '#4e545d', '#5f666f', '#737a84', '#8a919a',
                                '#a2a9b2', '#bbc1c8', '#d3d8dd', '#e9ecef', '#ffffff']),
            slope: [[0, '#f7f8fa'], [8, '#dcdfe4'], [16, '#bfc4cb'], [24, '#9ba1aa'], [32, '#767d87'],
                    [42, '#545b65'], [52, '#363c45'], [60, '#1b1f26']],
            roughness: [[0, '#f7f8fa'], [4, '#d8dce1'], [9, '#b4bac2'], [15, '#8d949d'], [22, '#676e78'],
                        [31, '#434952'], [40, '#1b1f26']],
        },
        dark: {
            // Lifted off black, which a dark page would swallow.
            elevation: ramp(H, ['#3a3f47', '#4a4f58', '#5c626b', '#70767f', '#868c95',
                                '#9ea4ad', '#b7bdc5', '#d0d5db', '#e8ebee', '#ffffff']),
            slope: [[0, '#2b3038'], [8, '#3d434c'], [16, '#525862'], [24, '#6a7079'], [32, '#858b94'],
                    [42, '#a3a9b1'], [52, '#c5cad0'], [60, '#f2f4f6']],
            roughness: [[0, '#2b3038'], [4, '#3f444d'], [9, '#565c65'], [15, '#71777f'], [22, '#8f959d'],
                        [31, '#b2b8bf'], [40, '#eef0f2']],
        },
    },
};

export const DEFAULT_PALETTE = 'imhof';
const KEY = 'nepaldtm-palette';

export function storedPalette() {
    const fromUrl = new URLSearchParams(location.search).get('palette');
    if (PALETTES[fromUrl]) return fromUrl;
    try {
        const saved = localStorage.getItem(KEY);
        if (PALETTES[saved]) return saved;
    } catch { /* storage blocked: the default will do */ }
    return DEFAULT_PALETTE;
}

export function rememberPalette(name) {
    try { localStorage.setItem(KEY, name); } catch { /* storage blocked: it lasts this page only */ }
}

// The stops for one measure, in the chosen scheme and theme.
export const stopsFor = (palette, theme, measure) =>
    (PALETTES[palette] ?? PALETTES[DEFAULT_PALETTE])[theme][measure];
