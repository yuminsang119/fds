/**
 * FDS Web Viewer - 3D Scan Renderer
 * Renders photogrammetry/LiDAR point clouds and meshes,
 * composited with FDS simulation data.
 */

class ScanRenderer {
    constructor(scene) {
        this.scene = scene;
        this.scanGroup = new THREE.Group();
        this.scanGroup.name = 'scan_data';
        this.scene.add(this.scanGroup);

        this.pointCloud = null;
        this.mesh = null;
        this.transform = null;
        this.scanData = null;

        // Display settings
        this.pointSize = 2.0;
        this.meshOpacity = 0.85;
        this.scanVisible = true;
        this.renderMode = 'points'; // 'points', 'mesh', 'wireframe'
    }

    /**
     * Load point cloud data from server response.
     */
    loadPointCloud(data) {
        this.clearScan();
        this.scanData = data;

        const positions = new Float32Array(data.positions);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        if (data.colors) {
            const colors = new Float32Array(data.colors);
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        if (data.normals) {
            const normals = new Float32Array(data.normals);
            geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
        }

        geometry.computeBoundingBox();

        const material = new THREE.PointsMaterial({
            size: this.pointSize,
            vertexColors: data.colors ? true : false,
            color: data.colors ? undefined : 0xaaaaaa,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9,
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scanGroup.add(this.pointCloud);

        return {
            numPoints: data.num_points,
            bounds: {
                min: data.bounds_min,
                max: data.bounds_max,
            }
        };
    }

    /**
     * Load triangle mesh data from server response.
     */
    loadMesh(data) {
        this.scanData = data;

        const positions = new Float32Array(data.positions);
        const indices = new Uint32Array(data.indices);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));

        if (data.normals) {
            geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(data.normals), 3));
        } else {
            geometry.computeVertexNormals();
        }

        if (data.colors) {
            geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(data.colors), 3));
        }

        if (data.uvs) {
            geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(data.uvs), 2));
        }

        // Solid mesh
        const meshMaterial = new THREE.MeshPhongMaterial({
            vertexColors: data.colors ? true : false,
            color: data.colors ? undefined : 0xcccccc,
            transparent: true,
            opacity: this.meshOpacity,
            side: THREE.DoubleSide,
            flatShading: false,
        });

        this.mesh = new THREE.Mesh(geometry, meshMaterial);
        this.scanGroup.add(this.mesh);

        // Wireframe overlay
        const wireGeo = new THREE.WireframeGeometry(geometry);
        const wireMat = new THREE.LineBasicMaterial({
            color: 0x444466,
            transparent: true,
            opacity: 0.15,
        });
        this.wireframe = new THREE.LineSegments(wireGeo, wireMat);
        this.wireframe.visible = false;
        this.scanGroup.add(this.wireframe);

        return {
            numVertices: data.num_vertices,
            numFaces: data.num_faces,
            bounds: {
                min: data.bounds_min,
                max: data.bounds_max,
            }
        };
    }

    /**
     * Apply spatial transform (from alignment).
     * matrix is column-major 4x4 from server.
     */
    applyTransform(transformData) {
        this.transform = transformData;

        if (transformData.matrix4x4) {
            const m = new THREE.Matrix4();
            m.fromArray(transformData.matrix4x4);
            this.scanGroup.applyMatrix4(m);
        } else {
            // Manual transform
            if (transformData.translation) {
                this.scanGroup.position.set(...transformData.translation);
            }
            if (transformData.scale !== undefined) {
                const s = transformData.scale;
                this.scanGroup.scale.set(s, s, s);
            }
        }
    }

    /**
     * Set manual position/rotation/scale from UI.
     */
    setManualTransform(params) {
        if (params.position) {
            this.scanGroup.position.set(params.position[0], params.position[1], params.position[2]);
        }
        if (params.rotation) {
            this.scanGroup.rotation.set(
                params.rotation[0] * Math.PI / 180,
                params.rotation[1] * Math.PI / 180,
                params.rotation[2] * Math.PI / 180
            );
        }
        if (params.scale !== undefined) {
            const s = params.scale;
            this.scanGroup.scale.set(s, s, s);
        }
    }

    /**
     * Set render mode.
     */
    setRenderMode(mode) {
        this.renderMode = mode;

        if (this.pointCloud) {
            this.pointCloud.visible = (mode === 'points');
        }
        if (this.mesh) {
            this.mesh.visible = (mode === 'mesh' || mode === 'wireframe');
            this.mesh.material.wireframe = (mode === 'wireframe');
        }
        if (this.wireframe) {
            this.wireframe.visible = (mode === 'mesh');
        }
    }

    /**
     * Set point size for point cloud rendering.
     */
    setPointSize(size) {
        this.pointSize = size;
        if (this.pointCloud) {
            this.pointCloud.material.size = size;
        }
    }

    /**
     * Set scan opacity (for blending with simulation).
     */
    setOpacity(opacity) {
        this.meshOpacity = opacity;
        if (this.pointCloud) {
            this.pointCloud.material.opacity = opacity;
        }
        if (this.mesh) {
            this.mesh.material.opacity = opacity;
        }
    }

    /**
     * Toggle scan visibility.
     */
    setVisible(visible) {
        this.scanVisible = visible;
        this.scanGroup.visible = visible;
    }

    /**
     * Add alignment markers to scene.
     */
    addMarker(position, color, label) {
        const geometry = new THREE.SphereGeometry(0.05, 16, 16);
        const material = new THREE.MeshBasicMaterial({ color: color });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(position[0], position[1], position[2]);
        sphere.userData = { type: 'marker', label: label };
        this.scanGroup.add(sphere);

        // Label sprite
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(0, 0, 128, 64);
        ctx.font = '24px monospace';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.fillText(label, 64, 40);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.set(position[0], position[1] + 0.15, position[2]);
        sprite.scale.set(0.3, 0.15, 1);
        this.scanGroup.add(sprite);

        return sphere;
    }

    /**
     * Get current transform as JSON for saving.
     */
    getTransformJSON() {
        return {
            position: this.scanGroup.position.toArray(),
            rotation: [
                this.scanGroup.rotation.x * 180 / Math.PI,
                this.scanGroup.rotation.y * 180 / Math.PI,
                this.scanGroup.rotation.z * 180 / Math.PI,
            ],
            scale: this.scanGroup.scale.x,
        };
    }

    /**
     * Clear all scan objects.
     */
    clearScan() {
        while (this.scanGroup.children.length > 0) {
            const child = this.scanGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
            this.scanGroup.remove(child);
        }
        this.pointCloud = null;
        this.mesh = null;
        this.wireframe = null;
        this.scanData = null;
    }

    dispose() {
        this.clearScan();
        this.scene.remove(this.scanGroup);
    }
}
