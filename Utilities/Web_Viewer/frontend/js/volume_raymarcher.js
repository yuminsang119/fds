/**
 * FDS Volume Ray Marcher
 * GPU shader-based volume rendering for realistic smoke/fire visualization.
 * Uses Three.js custom shaders with ray marching through 3D textures.
 */

const VolumeShaders = {

    vertexShader: `
        varying vec3 vOrigin;
        varying vec3 vDirection;
        uniform vec3 cameraPos;

        void main() {
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            vDirection = position - cameraPos;
            vOrigin = cameraPos;
            gl_Position = projectionMatrix * mvPosition;
        }
    `,

    fragmentShader: `
        precision highp float;
        precision highp sampler3D;

        varying vec3 vOrigin;
        varying vec3 vDirection;

        uniform sampler3D uVolume;
        uniform float uStepSize;
        uniform int uMaxSteps;
        uniform float uDensityScale;
        uniform float uOpacity;
        uniform float uTime;
        uniform vec3 uBoundsMin;
        uniform vec3 uBoundsMax;
        uniform vec3 uLightDir;
        uniform vec3 uLightColor;
        uniform float uLightIntensity;
        uniform int uColorMode; // 0=fire, 1=smoke, 2=thermal
        uniform float uAbsorption;
        uniform float uTemperatureScale;
        uniform float uBlackbodyIntensity;

        // --- Ray-AABB intersection ---
        vec2 intersectBox(vec3 orig, vec3 dir, vec3 boxMin, vec3 boxMax) {
            vec3 invDir = 1.0 / dir;
            vec3 tMin = (boxMin - orig) * invDir;
            vec3 tMax = (boxMax - orig) * invDir;
            vec3 t1 = min(tMin, tMax);
            vec3 t2 = max(tMin, tMax);
            float tNear = max(max(t1.x, t1.y), t1.z);
            float tFar = min(min(t2.x, t2.y), t2.z);
            return vec2(tNear, tFar);
        }

        // --- Fire colormap (blackbody radiation approximation) ---
        vec3 fireColor(float t) {
            // Physically-inspired blackbody: dark red -> orange -> yellow -> white
            vec3 c;
            t = clamp(t, 0.0, 1.0);
            c.r = smoothstep(0.0, 0.33, t);
            c.g = smoothstep(0.15, 0.6, t) * 0.9;
            c.b = smoothstep(0.5, 0.95, t) * 0.7;
            // Extra glow
            c += vec3(1.0, 0.4, 0.05) * pow(t, 3.0) * 2.0;
            return clamp(c, 0.0, 1.0);
        }

        // --- Smoke color (cool gray with depth) ---
        vec3 smokeColor(float density) {
            float t = clamp(density, 0.0, 1.0);
            vec3 darkSmoke = vec3(0.15, 0.15, 0.18);
            vec3 lightSmoke = vec3(0.6, 0.6, 0.65);
            return mix(darkSmoke, lightSmoke, t);
        }

        // --- Thermal colormap (blue->cyan->green->yellow->red) ---
        vec3 thermalColor(float t) {
            t = clamp(t, 0.0, 1.0);
            vec3 c;
            if (t < 0.25) {
                c = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 1.0), t * 4.0);
            } else if (t < 0.5) {
                c = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 1.0, 0.2), (t - 0.25) * 4.0);
            } else if (t < 0.75) {
                c = mix(vec3(0.0, 1.0, 0.2), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
            } else {
                c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
            }
            return c;
        }

        // --- Sample volume gradient (forward-difference, 3 samples vs 6) ---
        vec3 volumeGradient(vec3 p, float step) {
            float center = texture(uVolume, p).r;
            float dx = texture(uVolume, p + vec3(step, 0, 0)).r - center;
            float dy = texture(uVolume, p + vec3(0, step, 0)).r - center;
            float dz = texture(uVolume, p + vec3(0, 0, step)).r - center;
            return normalize(vec3(dx, dy, dz) + 1e-6);
        }

        // --- Light marching (shadow ray, optimized: 6 samples) ---
        float lightMarch(vec3 pos, vec3 lightDir, int steps) {
            float totalDensity = 0.0;
            float stepSize = 0.035;  // larger steps for shadow (was 0.02)
            vec3 p = pos;
            vec3 invBounds = 1.0 / (uBoundsMax - uBoundsMin);
            for (int i = 0; i < 6; i++) {
                if (i >= steps) break;
                p += lightDir * stepSize;
                vec3 tc = (p - uBoundsMin) * invBounds;
                if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) break;
                totalDensity += texture(uVolume, tc).r * stepSize * uDensityScale;
            }
            return exp(-totalDensity * uAbsorption);
        }

        void main() {
            vec3 rayDir = normalize(vDirection);
            vec2 tHit = intersectBox(vOrigin, rayDir, uBoundsMin, uBoundsMax);

            if (tHit.x > tHit.y) {
                discard;
                return;
            }

            tHit.x = max(tHit.x, 0.0);
            float stepSize = uStepSize;

            // Jitter start position to reduce banding
            float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
            tHit.x += stepSize * jitter;

            // --- Ray marching ---
            vec4 accumulated = vec4(0.0);
            float transmittance = 1.0;

            vec3 invBoundsSize = 1.0 / (uBoundsMax - uBoundsMin);

            for (int i = 0; i < 512; i++) {
                if (i >= uMaxSteps) break;
                if (transmittance < 0.01) break;  // early termination

                float t = tHit.x + float(i) * stepSize;
                if (t > tHit.y) break;

                vec3 pos = vOrigin + rayDir * t;

                // Convert world position to texture coordinates [0, 1]
                vec3 texCoord = (pos - uBoundsMin) * invBoundsSize;

                // Sample volume
                float density = texture(uVolume, texCoord).r;

                // Empty space skipping
                if (density < 0.005) {
                    // Skip ahead in empty regions
                    continue;
                }

                if (density > 0.01) {
                    float scaledDensity = density * uDensityScale;

                    // Color based on mode
                    vec3 color;
                    if (uColorMode == 0) {
                        // Fire mode: density maps to temperature/emission
                        color = fireColor(density * uTemperatureScale);
                        // Self-emission (fire glows)
                        color *= (1.0 + density * uBlackbodyIntensity * 3.0);
                    } else if (uColorMode == 1) {
                        // Smoke mode: scattering + shadows
                        color = smokeColor(density);

                        // Light scattering
                        float shadow = lightMarch(pos, normalize(uLightDir), 12);
                        vec3 grad = volumeGradient(texCoord, 0.01);
                        float diffuse = max(dot(grad, normalize(uLightDir)), 0.0);

                        color *= uLightColor * uLightIntensity * shadow * (0.3 + 0.7 * diffuse);
                        // Ambient
                        color += smokeColor(density) * 0.15;
                    } else {
                        // Thermal mode
                        color = thermalColor(density * uTemperatureScale);
                    }

                    // Absorption
                    float alpha = 1.0 - exp(-scaledDensity * stepSize * uAbsorption);
                    alpha *= uOpacity;

                    // Front-to-back compositing
                    accumulated.rgb += color * alpha * transmittance;
                    accumulated.a += alpha * transmittance;
                    transmittance *= (1.0 - alpha);
                }
            }

            // Background blend
            vec3 bgColor = vec3(0.05, 0.05, 0.1);
            vec3 finalColor = accumulated.rgb + bgColor * transmittance;

            gl_FragColor = vec4(finalColor, 1.0 - transmittance + 0.01);
        }
    `
};


class VolumeRayMarcher {
    constructor(scene, bounds) {
        this.scene = scene;
        this.bounds = bounds;
        this.mesh = null;
        this.volumeTexture = null;
        this.uniforms = null;
        this.colorMode = 0; // 0=fire, 1=smoke, 2=thermal
    }

    /**
     * Create 3D volume texture from flat data array.
     */
    createVolumeTexture(data, nx, ny, nz, maxVal) {
        const floatData = new Float32Array(nx * ny * nz);

        // Single-pass normalize (use server-provided maxVal if available)
        if (maxVal === undefined || maxVal <= 0) {
            maxVal = 0;
            for (let i = 0; i < data.length; i++) {
                if (data[i] > maxVal) maxVal = data[i];
            }
        }
        this._cachedMaxVal = maxVal;
        const scale = maxVal > 0 ? 1.0 / maxVal : 1.0;
        for (let i = 0; i < data.length; i++) {
            floatData[i] = data[i] * scale;
        }

        // Three.js Data3DTexture (r128+)
        if (typeof THREE.Data3DTexture !== 'undefined') {
            // Newer Three.js
            const tex = new THREE.Data3DTexture(floatData, nx, ny, nz);
            tex.format = THREE.RedFormat;
            tex.type = THREE.FloatType;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.wrapS = THREE.ClampToEdgeWrapping;
            tex.wrapT = THREE.ClampToEdgeWrapping;
            tex.wrapR = THREE.ClampToEdgeWrapping;
            tex.needsUpdate = true;
            return tex;
        }

        // Fallback: DataTexture3D
        if (typeof THREE.DataTexture3D !== 'undefined') {
            const tex = new THREE.DataTexture3D(floatData, nx, ny, nz);
            tex.format = THREE.RedFormat;
            tex.type = THREE.FloatType;
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.needsUpdate = true;
            return tex;
        }

        console.warn('3D textures not supported in this Three.js version');
        return null;
    }

    /**
     * Initialize the volume renderer with data.
     */
    init(data, nx, ny, nz, maxVal) {
        // Remove old mesh
        if (this.mesh) {
            this.scene.remove(this.mesh);
            if (this.volumeTexture) this.volumeTexture.dispose();
        }

        this.volumeTexture = this.createVolumeTexture(data, nx, ny, nz, maxVal);
        if (!this.volumeTexture) return false;

        const b = this.bounds;
        const size = [b.x[1] - b.x[0], b.z[1] - b.z[0], b.y[1] - b.y[0]];
        const center = [
            (b.x[0] + b.x[1]) / 2,
            (b.z[0] + b.z[1]) / 2,
            (b.y[0] + b.y[1]) / 2
        ];

        this.uniforms = {
            uVolume: { value: this.volumeTexture },
            uStepSize: { value: 0.005 },
            uMaxSteps: { value: 256 },
            uDensityScale: { value: 5.0 },
            uOpacity: { value: 0.8 },
            uTime: { value: 0.0 },
            uBoundsMin: { value: new THREE.Vector3(b.x[0], b.z[0], b.y[0]) },
            uBoundsMax: { value: new THREE.Vector3(b.x[1], b.z[1], b.y[1]) },
            uLightDir: { value: new THREE.Vector3(1, 1, 0.5).normalize() },
            uLightColor: { value: new THREE.Vector3(1.0, 0.95, 0.8) },
            uLightIntensity: { value: 2.0 },
            uColorMode: { value: 0 },
            uAbsorption: { value: 40.0 },
            uTemperatureScale: { value: 1.5 },
            uBlackbodyIntensity: { value: 2.0 },
            cameraPos: { value: new THREE.Vector3() },
        };

        const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
        const material = new THREE.ShaderMaterial({
            vertexShader: VolumeShaders.vertexShader,
            fragmentShader: VolumeShaders.fragmentShader,
            uniforms: this.uniforms,
            transparent: true,
            side: THREE.BackSide,
            depthWrite: false,
        });

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.set(center[0], center[1], center[2]);
        this.scene.add(this.mesh);

        return true;
    }

    /**
     * Update volume data for animation.
     */
    updateData(data, nx, ny, nz, maxVal) {
        if (!this.volumeTexture) {
            return this.init(data, nx, ny, nz, maxVal);
        }

        const floatData = new Float32Array(nx * ny * nz);
        // Use server-provided maxVal to skip finding max
        if (maxVal === undefined || maxVal <= 0) {
            maxVal = 0;
            for (let i = 0; i < data.length; i++) {
                if (data[i] > maxVal) maxVal = data[i];
            }
        }
        this._cachedMaxVal = maxVal;
        const scale = maxVal > 0 ? 1.0 / maxVal : 1.0;
        for (let i = 0; i < data.length; i++) {
            floatData[i] = data[i] * scale;
        }

        // Update texture data in-place
        this.volumeTexture.image = { data: floatData, width: nx, height: ny, depth: nz };
        this.volumeTexture.needsUpdate = true;
    }

    /**
     * Update uniforms each frame.
     */
    update(camera, deltaTime) {
        if (!this.uniforms) return;

        this.uniforms.cameraPos.value.copy(camera.position);
        this.uniforms.uTime.value += deltaTime;
    }

    /**
     * Set rendering parameters.
     */
    setParams(params) {
        if (!this.uniforms) return;

        if (params.density !== undefined) this.uniforms.uDensityScale.value = params.density;
        if (params.opacity !== undefined) this.uniforms.uOpacity.value = params.opacity;
        if (params.steps !== undefined) this.uniforms.uMaxSteps.value = params.steps;
        if (params.stepSize !== undefined) this.uniforms.uStepSize.value = params.stepSize;
        if (params.absorption !== undefined) this.uniforms.uAbsorption.value = params.absorption;
        if (params.temperatureScale !== undefined) this.uniforms.uTemperatureScale.value = params.temperatureScale;
        if (params.blackbodyIntensity !== undefined) this.uniforms.uBlackbodyIntensity.value = params.blackbodyIntensity;
        if (params.colorMode !== undefined) {
            this.uniforms.uColorMode.value = params.colorMode;
            this.colorMode = params.colorMode;
        }
        if (params.lightDir) {
            this.uniforms.uLightDir.value.set(...params.lightDir).normalize();
        }
        if (params.lightIntensity !== undefined) {
            this.uniforms.uLightIntensity.value = params.lightIntensity;
        }
    }

    /**
     * Remove from scene and free resources.
     */
    dispose() {
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
        }
        if (this.volumeTexture) {
            this.volumeTexture.dispose();
        }
    }
}
