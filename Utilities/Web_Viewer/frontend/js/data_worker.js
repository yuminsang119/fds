/**
 * FDS Web Viewer - Data Processing Web Worker
 * Offloads heavy computation from main thread to maintain 60fps UI.
 */

// Colormap LUTs (pre-computed on first use)
const colormapLUTs = {};

function generateFireLUT(size) {
    const lut = new Float32Array(size * 3);
    for (let i = 0; i < size; i++) {
        const t = i / (size - 1);
        let r, g, b;
        if (t < 0.25) {
            const s = t / 0.25;
            r = s * 0.6; g = 0; b = 0;
        } else if (t < 0.5) {
            const s = (t - 0.25) / 0.25;
            r = 0.6 + s * 0.4; g = s * 0.3; b = 0;
        } else if (t < 0.75) {
            const s = (t - 0.5) / 0.25;
            r = 1.0; g = 0.3 + s * 0.5; b = 0;
        } else {
            const s = (t - 0.75) / 0.25;
            r = 1.0; g = 0.8 + s * 0.2; b = s;
        }
        lut[i * 3] = r;
        lut[i * 3 + 1] = g;
        lut[i * 3 + 2] = b;
    }
    return lut;
}

function getLUT(name, size) {
    const key = `${name}_${size}`;
    if (!colormapLUTs[key]) {
        colormapLUTs[key] = generateFireLUT(size);
    }
    return colormapLUTs[key];
}

/**
 * Normalize volume data. Uses maxVal if provided to skip scanning.
 */
function normalizeVolume(data, maxVal) {
    const out = new Float32Array(data.length);
    if (!maxVal || maxVal <= 0) {
        maxVal = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] > maxVal) maxVal = data[i];
        }
    }
    const scale = maxVal > 0 ? 1.0 / maxVal : 1.0;
    for (let i = 0; i < data.length; i++) {
        out[i] = data[i] * scale;
    }
    return { data: out, maxVal: maxVal };
}

/**
 * Filter volume points above threshold and apply colormap.
 * Returns positions and colors as Float32Arrays.
 */
function filterVolumePoints(params) {
    const { data, nx, ny, nz, maxVal, threshold, bounds } = params;
    const invMaxVal = maxVal > 0 ? 1.0 / maxVal : 1.0;
    const lut = getLUT('fire', 256);

    const dx = (bounds.x[1] - bounds.x[0]) / nx;
    const dy = (bounds.y[1] - bounds.y[0]) / ny;
    const dz = (bounds.z[1] - bounds.z[0]) / nz;

    // First pass: count points
    let count = 0;
    for (let i = 0; i < data.length; i++) {
        if (data[i] * invMaxVal > threshold) count++;
    }

    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    let idx = 0;

    for (let iz = 0; iz < nz; iz++) {
        const y = bounds.z[0] + (iz + 0.5) * dz;
        const izOff = iz * ny * nx;
        for (let iy = 0; iy < ny; iy++) {
            const z = bounds.y[0] + (iy + 0.5) * dy;
            const iyOff = izOff + iy * nx;
            for (let ix = 0; ix < nx; ix++) {
                const val = data[iyOff + ix] * invMaxVal;
                if (val > threshold) {
                    const p3 = idx * 3;
                    positions[p3] = bounds.x[0] + (ix + 0.5) * dx;
                    positions[p3 + 1] = y;
                    positions[p3 + 2] = z;
                    const li = ((val * 255) | 0) * 3;
                    colors[p3] = lut[li];
                    colors[p3 + 1] = lut[li + 1];
                    colors[p3 + 2] = lut[li + 2];
                    idx++;
                }
            }
        }
    }

    return { positions, colors, count };
}

/**
 * Decode binary WebSocket frame.
 */
function decodeBinaryFrame(buffer) {
    const view = new DataView(buffer);
    const headerLen = view.getUint32(0, true);
    const headerStr = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLen));
    const header = JSON.parse(headerStr);
    const dataOffset = 4 + headerLen;
    const dataLen = (buffer.byteLength - dataOffset) / 4;
    const data = new Float32Array(buffer, dataOffset, dataLen);
    return { header, data };
}

// Message handler
self.onmessage = function(e) {
    const { type, id } = e.data;

    switch (type) {
        case 'normalize_volume': {
            const { data, maxVal } = e.data;
            const result = normalizeVolume(data, maxVal);
            self.postMessage(
                { type: 'volume_normalized', id, data: result.data, maxVal: result.maxVal },
                [result.data.buffer]
            );
            break;
        }
        case 'filter_points': {
            const result = filterVolumePoints(e.data);
            self.postMessage(
                { type: 'points_filtered', id, ...result },
                [result.positions.buffer, result.colors.buffer]
            );
            break;
        }
        case 'decode_binary': {
            const result = decodeBinaryFrame(e.data.buffer);
            self.postMessage(
                { type: 'binary_decoded', id, ...result },
                [result.data.buffer]
            );
            break;
        }
        default:
            self.postMessage({ type: 'error', message: `Unknown type: ${type}` });
    }
};
