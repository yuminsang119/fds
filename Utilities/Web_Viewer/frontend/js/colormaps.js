/**
 * FDS Web Viewer - Colormaps
 * Color mapping functions for temperature/density visualization.
 */

const Colormaps = {
    fire: function(t) {
        // Black -> Red -> Orange -> Yellow -> White
        t = Math.max(0, Math.min(1, t));
        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = s * 0.6;
            g = 0;
            b = 0;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0.6 + s * 0.4;
            g = s * 0.3;
            b = 0;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = 1.0;
            g = 0.3 + s * 0.5;
            b = 0;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1.0;
            g = 0.8 + s * 0.2;
            b = s;
        }
        return [r, g, b];
    },

    thermal: function(t) {
        // Blue -> Cyan -> Green -> Yellow -> Red
        t = Math.max(0, Math.min(1, t));
        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = 0; g = 0; b = 0.5 + s * 0.5;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0; g = s; b = 1.0 - s * 0.5;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = s; g = 1.0; b = 0.5 - s * 0.5;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1.0; g = 1.0 - s; b = 0;
        }
        return [r, g, b];
    },

    rainbow: function(t) {
        t = Math.max(0, Math.min(1, t));
        const h = (1.0 - t) * 270 / 360;
        const s = 1.0, v = 1.0;
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const u = v * (1 - (1 - f) * s);
        switch (i % 6) {
            case 0: return [v, u, p];
            case 1: return [q, v, p];
            case 2: return [p, v, u];
            case 3: return [p, q, v];
            case 4: return [u, p, v];
            case 5: return [v, p, q];
        }
        return [1, 1, 1];
    },

    grayscale: function(t) {
        t = Math.max(0, Math.min(1, t));
        return [t, t, t];
    },

    /**
     * Create a colorbar canvas gradient for a given colormap.
     */
    createColorbar: function(colormapName, canvas) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        const gradient = ctx.createLinearGradient(0, 0, w, 0);
        const fn = this[colormapName] || this.fire;
        for (let i = 0; i <= 10; i++) {
            const t = i / 10;
            const [r, g, b] = fn(t);
            gradient.addColorStop(t, `rgb(${r*255|0},${g*255|0},${b*255|0})`);
        }
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);
    }
};
