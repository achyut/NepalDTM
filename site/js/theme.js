// Light by default. The switcher sets data-theme="dark" on <html> and remembers the choice in this browser;
// an inline script in each page's <head> applies the saved choice before first paint.

const KEY = 'nepaldtm-theme';
const root = document.documentElement;

export const currentTheme = () => root.dataset.theme === 'dark' ? 'dark' : 'light';

export function initThemeToggle(onChange) {
    const buttons = document.querySelectorAll('[data-theme-toggle]');
    const sync = () => buttons.forEach(b => b.setAttribute('aria-label',
        currentTheme() === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'));
    sync();
    buttons.forEach(b => b.addEventListener('click', () => {
        const next = currentTheme() === 'dark' ? 'light' : 'dark';
        if (next === 'dark') root.dataset.theme = 'dark';
        else delete root.dataset.theme;
        try { localStorage.setItem(KEY, next); } catch { /* storage blocked: the choice lasts this page only */ }
        sync();
        onChange?.(next);
    }));
}
