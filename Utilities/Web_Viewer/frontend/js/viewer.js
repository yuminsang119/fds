/**
 * FDS Web Viewer - Main 3D Viewer
 * Three.js based visualization for FDS simulation results.
 * Supports 2D slice, 3D point cloud, and GPU ray marching volume rendering.
 */

class FDSViewer {
    constructor() {
        this.canvas = document.getElementById('render-canvas');
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Data
        this.sliceFrames = [];
        this.volumeFrames = [];
        this.metadata = null;
        this.currentFrame = 0;
        this.viewMode = 'slice'; // 'slice', 'volume', or 'raymarch'

        // Playback
        this.isPlaying = false;
        this.playbackSpeed = 1.0;
        this.lastFrameTime = 0;
        this.lastAnimTime = 0;

        // Scene objects
        this.sliceMesh = null;
        this.volumeGroup = null;
        this.gridHelper = null;
        this.axesHelper = null;
        this.boundingBox = null;

        // Ray marching renderer
        this.rayMarcher = null;

        // Settings
        this.opacity = 0.8;
        this.colormapName = 'fire';
        this.showGrid = true;
        this.showAxes = true;
        this.showWireframe = false;

        // FPS
        this.frameCount = 0;
        this.fpsTime = performance.now();

        this.init();
        this.setupEventListeners();
        this.checkGPUStatus();
        this.animate();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0d0d1a);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
        this.camera.position.set(8, 6, 8);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.target.set(2.5, 2.5, 1.5);
        this.controls.update();

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 10, 10);
        this.scene.add(dirLight);

        // Grid
        this.gridHelper = new THREE.GridHelper(10, 20, 0x333355, 0x222244);
        this.scene.add(this.gridHelper);

        // Axes
        this.axesHelper = new THREE.AxesHelper(2);
        this.scene.add(this.axesHelper);

        // Resize handler
        window.addEventListener('resize', () => this.onResize());
        this.onResize();
    }

    setupEventListeners() {
        // Load Demo
        document.getElementById('btn-load-demo').addEventListener('click', () => {
            this.loadDemo();
        });

        // View mode
        document.querySelectorAll('[data-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-mode]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.viewMode = e.target.dataset.mode;
                this.currentFrame = 0;
                this.clearSceneObjects();
                this.updateView();

                // Show/hide ray marching controls
                const rmControls = document.getElementById('raymarch-controls');
                if (rmControls) {
                    rmControls.style.display = this.viewMode === 'raymarch' ? 'block' : 'none';
                }
            });
        });

        // Playback
        document.getElementById('btn-play').addEventListener('click', () => this.togglePlay());
        document.getElementById('btn-prev').addEventListener('click', () => this.prevFrame());
        document.getElementById('btn-next').addEventListener('click', () => this.nextFrame());

        // Frame slider
        document.getElementById('frame-slider').addEventListener('input', (e) => {
            this.currentFrame = parseInt(e.target.value);
            this.updateView();
        });

        // Speed slider
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speed-label').textContent = this.playbackSpeed.toFixed(1) + 'x';
        });

        // Opacity
        document.getElementById('opacity-slider').addEventListener('input', (e) => {
            this.opacity = parseFloat(e.target.value);
            if (this.rayMarcher) {
                this.rayMarcher.setParams({ opacity: this.opacity });
            }
            this.updateView();
        });

        // Display toggles
        document.getElementById('chk-grid').addEventListener('change', (e) => {
            this.showGrid = e.target.checked;
            this.gridHelper.visible = this.showGrid;
        });

        document.getElementById('chk-axes').addEventListener('change', (e) => {
            this.showAxes = e.target.checked;
            this.axesHelper.visible = this.showAxes;
        });

        document.getElementById('chk-wireframe').addEventListener('change', (e) => {
            this.showWireframe = e.target.checked;
            this.updateView();
        });

        // Colormap
        document.getElementById('colormap-select').addEventListener('change', (e) => {
            this.colormapName = e.target.value;
            this.updateColorbar();
            this.updateView();
        });

        // Ray Marching controls
        this.setupRayMarchControls();
    }

    setupRayMarchControls() {
        const densitySlider = document.getElementById('density-slider');
        const stepsSlider = document.getElementById('steps-slider');
        const absorptionSlider = document.getElementById('absorption-slider');
        const lightSlider = document.getElementById('light-slider');
        const renderModeSelect = document.getElementById('render-mode-select');

        if (densitySlider) {
            densitySlider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                document.getElementById('density-label').textContent = val.toFixed(1);
                if (this.rayMarcher) this.rayMarcher.setParams({ density: val });
            });
        }

        if (stepsSlider) {
            stepsSlider.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                document.getElementById('steps-label').textContent = val;
                if (this.rayMarcher) this.rayMarcher.setParams({ steps: val });
            });
        }

        if (absorptionSlider) {
            absorptionSlider.addEventListener('input', (e) => {
                const val = parseInt(e.target.value);
                document.getElementById('absorption-label').textContent = val;
                if (this.rayMarcher) this.rayMarcher.setParams({ absorption: val });
            });
        }

        if (lightSlider) {
            lightSlider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                document.getElementById('light-label').textContent = val.toFixed(1);
                if (this.rayMarcher) this.rayMarcher.setParams({ lightIntensity: val });
            });
        }

        if (renderModeSelect) {
            renderModeSelect.addEventListener('change', (e) => {
                const mode = parseInt(e.target.value);
                if (this.rayMarcher) this.rayMarcher.setParams({ colorMode: mode });
            });
        }
    }

    async checkGPUStatus() {
        try {
            const res = await fetch('/api/gpu/status');
            const data = await res.json();
            const el = document.getElementById('gpu-status');
            if (el) {
                if (data.available && data.renderer && data.renderer.num_gpus > 0) {
                    const gpus = data.renderer.gpus;
                    el.textContent = `${gpus.length}x ${gpus[0].name}`;
                    el.style.color = '#2ecc71';
                } else {
                    el.textContent = 'WebGL (Client)';
                    el.style.color = '#f39c12';
                }
            }
        } catch {
            const el = document.getElementById('gpu-status');
            if (el) {
                el.textContent = 'WebGL (Client)';
                el.style.color = '#f39c12';
            }
        }
    }

    clearSceneObjects() {
        if (this.sliceMesh) {
            this.scene.remove(this.sliceMesh);
            this.sliceMesh.geometry.dispose();
            this.sliceMesh.material.dispose();
            this.sliceMesh = null;
        }
        if (this.volumeGroup) {
            this.scene.remove(this.volumeGroup);
            this.volumeGroup = null;
        }
        if (this.rayMarcher) {
            this.rayMarcher.dispose();
            this.rayMarcher = null;
        }
    }

    async loadDemo() {
        this.showLoading('Loading demo simulation data...');
        this.setStatus('Loading...', 'status-loading');

        try {
            // Fetch metadata
            const metaRes = await fetch('/api/demo');
            this.metadata = await metaRes.json();

            document.getElementById('sim-title').textContent = this.metadata.title;

            // Fetch all slice frames
            this.showLoading('Loading slice frames...');
            this.sliceFrames = [];
            for (let i = 0; i < this.metadata.n_slice_frames; i++) {
                const res = await fetch(`/api/demo/slice/${i}`);
                this.sliceFrames.push(await res.json());
                if (i % 10 === 0) {
                    this.showLoading(`Loading slices... ${i}/${this.metadata.n_slice_frames}`);
                }
            }

            // Fetch volume frames
            this.showLoading('Loading volume frames...');
            this.volumeFrames = [];
            for (let i = 0; i < this.metadata.n_volume_frames; i++) {
                const res = await fetch(`/api/demo/volume/${i}`);
                this.volumeFrames.push(await res.json());
            }

            // Setup scene
            this.setupBoundingBox();
            this.updateColorbar();

            // Set slider
            this.updateSliderMax();

            this.currentFrame = 0;
            this.updateView();
            this.hideLoading();
            this.setStatus('Ready', 'status-ready');

        } catch (err) {
            console.error('Failed to load demo:', err);
            this.hideLoading();
            this.setStatus('Error', 'status-idle');
        }
    }

    updateSliderMax() {
        let maxFrame = 0;
        if (this.viewMode === 'slice') {
            maxFrame = this.sliceFrames.length - 1;
        } else {
            maxFrame = this.volumeFrames.length - 1;
        }
        document.getElementById('frame-slider').max = Math.max(0, maxFrame);
    }

    setupBoundingBox() {
        if (this.boundingBox) this.scene.remove(this.boundingBox);

        const b = this.metadata.bounds;
        const dx = b.x[1] - b.x[0];
        const dy = b.y[1] - b.y[0];
        const dz = b.z[1] - b.z[0];

        const geometry = new THREE.BoxGeometry(dx, dz, dy);
        const edges = new THREE.EdgesGeometry(geometry);
        this.boundingBox = new THREE.LineSegments(
            edges,
            new THREE.LineBasicMaterial({ color: 0x4466aa, linewidth: 1 })
        );
        this.boundingBox.position.set(
            b.x[0] + dx / 2,
            b.z[0] + dz / 2,
            b.y[0] + dy / 2
        );
        this.scene.add(this.boundingBox);

        // Update camera target
        this.controls.target.set(
            b.x[0] + dx / 2,
            b.z[0] + dz / 2,
            b.y[0] + dy / 2
        );
        this.controls.update();
    }

    updateView() {
        if (this.viewMode === 'slice') {
            this.updateSliceView();
        } else if (this.viewMode === 'volume') {
            this.updateVolumeView();
        } else if (this.viewMode === 'raymarch') {
            this.updateRayMarchView();
        }

        this.updateSliderMax();
        document.getElementById('frame-slider').value = this.currentFrame;
    }

    updateSliceView() {
        // Remove volume objects
        if (this.volumeGroup) {
            this.scene.remove(this.volumeGroup);
            this.volumeGroup = null;
        }
        if (this.rayMarcher) {
            this.rayMarcher.dispose();
            this.rayMarcher = null;
        }

        if (!this.sliceFrames.length) return;

        const frame = this.sliceFrames[this.currentFrame] || this.sliceFrames[0];
        const nx = frame.nx;
        const ny = frame.ny;
        const data = frame.data;
        const minVal = frame.min_val;
        const maxVal = frame.max_val;
        const range = maxVal - minVal || 1;

        // Create texture from data
        const texData = new Uint8Array(nx * ny * 4);
        const colormapFn = Colormaps[this.colormapName] || Colormaps.fire;

        for (let i = 0; i < nx * ny; i++) {
            const t = (data[i] - minVal) / range;
            const [r, g, b] = colormapFn(t);
            texData[i * 4] = (r * 255) | 0;
            texData[i * 4 + 1] = (g * 255) | 0;
            texData[i * 4 + 2] = (b * 255) | 0;
            texData[i * 4 + 3] = (this.opacity * 255) | 0;
        }

        const texture = new THREE.DataTexture(texData, nx, ny, THREE.RGBAFormat);
        texture.needsUpdate = true;
        texture.magFilter = THREE.LinearFilter;
        texture.minFilter = THREE.LinearFilter;

        if (this.sliceMesh) {
            this.sliceMesh.material.map = texture;
            this.sliceMesh.material.opacity = this.opacity;
            this.sliceMesh.material.wireframe = this.showWireframe;
            this.sliceMesh.material.needsUpdate = true;
        } else {
            const b = this.metadata.bounds;
            const geometry = new THREE.PlaneGeometry(
                b.x[1] - b.x[0],
                b.y[1] - b.y[0]
            );
            const material = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: this.opacity,
                side: THREE.DoubleSide,
                wireframe: this.showWireframe,
            });
            this.sliceMesh = new THREE.Mesh(geometry, material);
            this.sliceMesh.rotation.x = -Math.PI / 2;
            this.sliceMesh.position.set(
                (b.x[0] + b.x[1]) / 2,
                frame.plane_value || 1.5,
                (b.y[0] + b.y[1]) / 2
            );
            this.scene.add(this.sliceMesh);
        }

        // Update UI
        document.getElementById('sim-time').textContent = frame.time.toFixed(2) + ' s';
        document.getElementById('data-min').textContent = minVal.toFixed(1) + ' ' + (frame.units || '');
        document.getElementById('data-max').textContent = maxVal.toFixed(1) + ' ' + (frame.units || '');
        document.getElementById('frame-info').textContent = `Frame: ${this.currentFrame + 1}/${this.sliceFrames.length}`;
    }

    updateVolumeView() {
        // Remove slice
        if (this.sliceMesh) {
            this.scene.remove(this.sliceMesh);
            this.sliceMesh = null;
        }
        if (this.rayMarcher) {
            this.rayMarcher.dispose();
            this.rayMarcher = null;
        }

        if (!this.volumeFrames.length) return;

        const frame = this.volumeFrames[this.currentFrame] || this.volumeFrames[0];
        const nx = frame.nx;
        const ny = frame.ny;
        const nz = frame.nz;
        const data = frame.data;
        const maxVal = frame.max_val || 1;

        // Remove old volume
        if (this.volumeGroup) {
            this.scene.remove(this.volumeGroup);
        }
        this.volumeGroup = new THREE.Group();

        const b = this.metadata.bounds;
        const dx = (b.x[1] - b.x[0]) / nx;
        const dy = (b.y[1] - b.y[0]) / ny;
        const dz = (b.z[1] - b.z[0]) / nz;
        const colormapFn = Colormaps[this.colormapName] || Colormaps.fire;

        const positions = [];
        const colors = [];
        const threshold = 0.05;

        for (let iz = 0; iz < nz; iz++) {
            for (let iy = 0; iy < ny; iy++) {
                for (let ix = 0; ix < nx; ix++) {
                    const idx = iz * ny * nx + iy * nx + ix;
                    const val = data[idx] / maxVal;

                    if (val > threshold) {
                        const x = b.x[0] + (ix + 0.5) * dx;
                        const y = b.z[0] + (iz + 0.5) * dz;
                        const z = b.y[0] + (iy + 0.5) * dy;

                        positions.push(x, y, z);
                        const [r, g, bb] = colormapFn(val);
                        colors.push(r, g, bb);
                    }
                }
            }
        }

        if (positions.length > 0) {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.PointsMaterial({
                size: Math.max(dx, dy, dz) * 2,
                vertexColors: true,
                transparent: true,
                opacity: this.opacity * 0.7,
                sizeAttenuation: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false,
            });

            const points = new THREE.Points(geometry, material);
            this.volumeGroup.add(points);
        }

        this.scene.add(this.volumeGroup);

        document.getElementById('sim-time').textContent = frame.time.toFixed(2) + ' s';
        document.getElementById('data-min').textContent = '0.00';
        document.getElementById('data-max').textContent = maxVal.toFixed(3);
        document.getElementById('frame-info').textContent = `Frame: ${this.currentFrame + 1}/${this.volumeFrames.length}`;
    }

    updateRayMarchView() {
        // Remove other renderers
        if (this.sliceMesh) {
            this.scene.remove(this.sliceMesh);
            this.sliceMesh = null;
        }
        if (this.volumeGroup) {
            this.scene.remove(this.volumeGroup);
            this.volumeGroup = null;
        }

        if (!this.volumeFrames.length || !this.metadata) return;

        const frame = this.volumeFrames[this.currentFrame] || this.volumeFrames[0];

        // Initialize ray marcher if needed
        if (!this.rayMarcher) {
            this.rayMarcher = new VolumeRayMarcher(this.scene, this.metadata.bounds);
        }

        // Update or init volume data
        if (this.rayMarcher.volumeTexture) {
            this.rayMarcher.updateData(frame.data, frame.nx, frame.ny, frame.nz);
        } else {
            const success = this.rayMarcher.init(frame.data, frame.nx, frame.ny, frame.nz);
            if (!success) {
                console.warn('Ray marching not supported, falling back to point cloud');
                this.viewMode = 'volume';
                this.updateVolumeView();
                return;
            }
        }

        // Apply current settings
        const renderMode = document.getElementById('render-mode-select');
        this.rayMarcher.setParams({
            opacity: this.opacity,
            colorMode: renderMode ? parseInt(renderMode.value) : 0,
            density: parseFloat(document.getElementById('density-slider')?.value || 5),
            steps: parseInt(document.getElementById('steps-slider')?.value || 256),
            absorption: parseInt(document.getElementById('absorption-slider')?.value || 40),
            lightIntensity: parseFloat(document.getElementById('light-slider')?.value || 2),
        });

        document.getElementById('sim-time').textContent = frame.time.toFixed(2) + ' s';
        document.getElementById('data-min').textContent = '0.00';
        document.getElementById('data-max').textContent = (frame.max_val || 0).toFixed(3);
        document.getElementById('frame-info').textContent = `Frame: ${this.currentFrame + 1}/${this.volumeFrames.length}`;
    }

    updateColorbar() {
        const container = document.getElementById('colorbar');
        container.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.width = 220;
        canvas.height = 16;
        canvas.style.width = '100%';
        canvas.style.height = '16px';
        canvas.style.borderRadius = '3px';
        Colormaps.createColorbar(this.colormapName, canvas);
        container.appendChild(canvas);
    }

    // Playback
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('btn-play');
        btn.innerHTML = this.isPlaying ? '&#9646;&#9646;' : '&#9654;';
    }

    nextFrame() {
        const frames = this.viewMode === 'slice' ? this.sliceFrames : this.volumeFrames;
        if (frames.length === 0) return;
        this.currentFrame = (this.currentFrame + 1) % frames.length;
        this.updateView();
    }

    prevFrame() {
        const frames = this.viewMode === 'slice' ? this.sliceFrames : this.volumeFrames;
        if (frames.length === 0) return;
        this.currentFrame = (this.currentFrame - 1 + frames.length) % frames.length;
        this.updateView();
    }

    // Animation loop
    animate() {
        requestAnimationFrame(() => this.animate());

        const now = performance.now();
        const deltaTime = (now - (this.lastAnimTime || now)) / 1000;
        this.lastAnimTime = now;

        // FPS counter
        this.frameCount++;
        if (now - this.fpsTime >= 1000) {
            document.getElementById('fps-counter').textContent = this.frameCount + ' FPS';
            this.frameCount = 0;
            this.fpsTime = now;
        }

        // Auto-play
        if (this.isPlaying && now - this.lastFrameTime > (100 / this.playbackSpeed)) {
            this.nextFrame();
            this.lastFrameTime = now;
        }

        // Update ray marcher camera position
        if (this.rayMarcher) {
            this.rayMarcher.update(this.camera, deltaTime);
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const container = document.getElementById('viewport');
        const w = container.clientWidth;
        const h = container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    // UI helpers
    showLoading(text) {
        document.getElementById('loading-overlay').classList.remove('hidden');
        document.getElementById('loading-text').textContent = text;
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    setStatus(text, cls) {
        const el = document.getElementById('sim-status');
        el.textContent = text;
        el.className = cls;
    }
}

// Initialize
const viewer = new FDSViewer();
