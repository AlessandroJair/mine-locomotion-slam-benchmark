#!/usr/bin/env python3
"""Point-cloud and mesh readers, and an approximate nearest-neighbour index.

Numpy only.  The workstation has neither scipy nor open3d, and requiring either
would mean the paper's map-accuracy numbers cannot be reproduced without first
installing packages, so everything needed is implemented here:

  read_pcd / read_ply   the two formats RTAB-Map exports (ASCII and binary)
  sample_stl / sample_dae   surface sampling of the world's collision meshes
  VoxelIndex            grid-hashed nearest-neighbour queries

VoxelIndex exists because a Chamfer distance between two clouds of 10^5-10^6
points is 10^10-10^12 brute-force distance evaluations.  The index buckets the
reference cloud into cubic cells of side `cell`, and a query searches its own
cell first, then rings of neighbouring cells outward, stopping as soon as the
best distance found is smaller than the distance to the next unexplored ring.
That test is what makes the result EXACT rather than approximate, up to the
`max_radius` cutoff.  Points with no neighbour inside max_radius are reported
separately rather than being silently assigned the cutoff value, which would
quietly bias the mean.
"""

import os
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np


# ---------------------------------------------------------------------------
# point clouds
# ---------------------------------------------------------------------------

def read_pcd(path):
    """PCD v0.7, ascii or binary (uncompressed).  Returns Nx3 float array."""
    with open(path, 'rb') as f:
        fields, sizes, types, counts = [], [], [], []
        npoints = 0
        fmt = 'ascii'
        while True:
            line = f.readline()
            if not line:
                raise ValueError('unexpected end of header in {}'.format(path))
            text = line.decode('ascii', 'replace').strip()
            if text.startswith('FIELDS'):
                fields = text.split()[1:]
            elif text.startswith('SIZE'):
                sizes = [int(v) for v in text.split()[1:]]
            elif text.startswith('TYPE'):
                types = text.split()[1:]
            elif text.startswith('COUNT'):
                counts = [int(v) for v in text.split()[1:]]
            elif text.startswith('POINTS'):
                npoints = int(text.split()[1])
            elif text.startswith('DATA'):
                fmt = text.split()[1]
                break

        if not counts:
            counts = [1] * len(fields)

        if fmt == 'ascii':
            data = np.loadtxt(f, dtype=np.float64, ndmin=2)
            idx = [fields.index(c) for c in ('x', 'y', 'z')]
            return data[:, idx]

        if fmt != 'binary':
            raise ValueError('PCD DATA {} is not supported (convert to ascii '
                             'or binary with pcl_convert_pcd_ascii_binary)'
                             .format(fmt))

        np_types = {('F', 4): 'f4', ('F', 8): 'f8',
                    ('U', 1): 'u1', ('U', 2): 'u2', ('U', 4): 'u4',
                    ('I', 1): 'i1', ('I', 2): 'i2', ('I', 4): 'i4'}
        dtype = []
        for name, t, s, c in zip(fields, types, sizes, counts):
            base = np_types.get((t, s))
            if base is None:
                raise ValueError('unsupported PCD field {} {}{}'.format(name, t, s))
            dtype.append((name, base, c) if c > 1 else (name, base))
        arr = np.frombuffer(f.read(), dtype=np.dtype(dtype), count=npoints)
        return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)


def read_ply(path):
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('{} is not a PLY file'.format(path))
        fmt = None
        props = []
        nvert = 0
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError('unexpected end of PLY header')
            text = line.decode('ascii', 'replace').strip()
            if text.startswith('format'):
                fmt = text.split()[1]
            elif text.startswith('element'):
                parts = text.split()
                in_vertex = parts[1] == 'vertex'
                if in_vertex:
                    nvert = int(parts[2])
            elif text.startswith('property') and in_vertex:
                parts = text.split()
                if parts[1] != 'list':
                    props.append((parts[2], parts[1]))
            elif text == 'end_header':
                break

        names = [n for n, _ in props]
        idx = [names.index(c) for c in ('x', 'y', 'z')]

        if fmt == 'ascii':
            rows = []
            for _ in range(nvert):
                rows.append([float(v) for v in f.readline().split()])
            return np.array(rows)[:, idx]

        ply_types = {'float': 'f4', 'float32': 'f4', 'double': 'f8',
                     'float64': 'f8', 'uchar': 'u1', 'uint8': 'u1',
                     'char': 'i1', 'int8': 'i1', 'short': 'i2', 'int16': 'i2',
                     'ushort': 'u2', 'uint16': 'u2', 'int': 'i4',
                     'int32': 'i4', 'uint': 'u4', 'uint32': 'u4'}
        endian = '<' if fmt == 'binary_little_endian' else '>'
        dtype = np.dtype([(n, endian + ply_types[t]) for n, t in props])
        arr = np.frombuffer(f.read(dtype.itemsize * nvert), dtype=dtype,
                            count=nvert)
        return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)


def read_cloud(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pcd':
        return read_pcd(path)
    if ext == '.ply':
        return read_ply(path)
    raise ValueError('unsupported cloud format: {}'.format(ext))


# ---------------------------------------------------------------------------
# meshes -> surface samples
# ---------------------------------------------------------------------------

def _sample_triangles(vertices, faces, density):
    """Area-weighted uniform sampling over a triangle soup.

    Sampling proportionally to area (rather than a fixed count per triangle)
    matters here: the mine mesh has a few enormous floor triangles and many
    tiny wall ones, and per-triangle sampling would put most of the reference
    points on the walls.
    """
    if len(faces) == 0:
        return np.zeros((0, 3))
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 0:
        return np.zeros((0, 3))

    n = max(1, int(total * density))
    rng = np.random.default_rng(0)
    probs = areas / total
    tri = rng.choice(len(faces), size=n, p=probs)
    u = rng.random(n)
    w = rng.random(n)
    over = u + w > 1.0
    u[over] = 1.0 - u[over]
    w[over] = 1.0 - w[over]
    return v0[tri] + u[:, None] * (v1[tri] - v0[tri]) + w[:, None] * (v2[tri] - v0[tri])


def load_stl(path):
    with open(path, 'rb') as f:
        head = f.read(84)
        if head[:5].lower() == b'solid' and b'facet' in head:
            f.seek(0)
            text = f.read().decode('ascii', 'replace')
            vals = re.findall(r'vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)',
                              text)
            v = np.array(vals, dtype=float)
            faces = np.arange(len(v)).reshape(-1, 3)
            return v, faces
        ntri = struct.unpack('<I', head[80:84])[0]
        data = np.frombuffer(f.read(50 * ntri), dtype=np.uint8)
        if len(data) < 50 * ntri:
            ntri = len(data) // 50
            data = data[:50 * ntri]
        data = data.reshape(ntri, 50)
        verts = np.frombuffer(data[:, 12:48].tobytes(),
                              dtype='<f4').reshape(ntri * 3, 3).astype(float)
        return verts, np.arange(ntri * 3).reshape(-1, 3)


_COLLADA_NS = '{http://www.collada.org/2005/11/COLLADASchema}'


def load_dae(path):
    """Vertices and triangles of a COLLADA file, in the file's own frame.

    Handles the two things that actually differ between exporters: the
    <up_axis> declaration and the <unit meter=...> scale.  Ignoring either one
    silently rotates or rescales the whole reference geometry, which would look
    like a catastrophic mapping error rather than a parsing bug.
    """
    root = ET.parse(path).getroot()

    def tag(name):
        return _COLLADA_NS + name

    asset = root.find(tag('asset'))
    up = 'Y_UP'
    scale = 1.0
    if asset is not None:
        u = asset.find(tag('up_axis'))
        if u is not None and u.text:
            up = u.text.strip()
        unit = asset.find(tag('unit'))
        if unit is not None and unit.get('meter'):
            scale = float(unit.get('meter'))

    all_v = []
    all_f = []
    offset = 0
    lib = root.find(tag('library_geometries'))
    if lib is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    for geom in lib.findall(tag('geometry')):
        mesh = geom.find(tag('mesh'))
        if mesh is None:
            continue

        sources = {}
        for src in mesh.findall(tag('source')):
            fa = src.find(tag('float_array'))
            if fa is None or not fa.text:
                continue
            sources['#' + src.get('id')] = np.array(
                fa.text.split(), dtype=float)

        verts_node = mesh.find(tag('vertices'))
        pos_src = None
        if verts_node is not None:
            for inp in verts_node.findall(tag('input')):
                if inp.get('semantic') == 'POSITION':
                    pos_src = inp.get('source')
            vert_id = '#' + verts_node.get('id')
        else:
            vert_id = None

        if pos_src is None or pos_src not in sources:
            continue
        V = sources[pos_src].reshape(-1, 3)

        for prim in list(mesh.findall(tag('triangles'))) + \
                list(mesh.findall(tag('polylist'))):
            inputs = prim.findall(tag('input'))
            stride = max(int(i.get('offset', 0)) for i in inputs) + 1
            vert_offset = None
            for i in inputs:
                if i.get('semantic') == 'VERTEX' and i.get('source') == vert_id:
                    vert_offset = int(i.get('offset', 0))
            if vert_offset is None:
                continue
            p = prim.find(tag('p'))
            if p is None or not p.text:
                continue
            idx = np.array(p.text.split(), dtype=int)
            vidx = idx[vert_offset::stride]

            if prim.tag == tag('polylist'):
                vcount_node = prim.find(tag('vcount'))
                counts = (np.array(vcount_node.text.split(), dtype=int)
                          if vcount_node is not None and vcount_node.text
                          else np.full(len(vidx) // 3, 3))
                faces = []
                pos = 0
                for c in counts:
                    poly = vidx[pos:pos + c]
                    for k in range(1, c - 1):      # fan triangulation
                        faces.append([poly[0], poly[k], poly[k + 1]])
                    pos += c
                faces = np.array(faces, dtype=int) if faces else np.zeros((0, 3), int)
            else:
                faces = vidx[:len(vidx) - len(vidx) % 3].reshape(-1, 3)

            if len(faces):
                all_v.append(V)
                all_f.append(faces + offset)
                offset += len(V)

    if not all_v:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    V = np.vstack(all_v) * scale
    F = np.vstack(all_f)

    if up.upper().startswith('Y'):
        # Y_UP -> Z_UP:  (x, y, z) -> (x, -z, y)
        V = np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])
    elif up.upper().startswith('X'):
        V = np.column_stack([-V[:, 1], V[:, 0], V[:, 2]])
    return V, F


def sample_mesh_file(path, density):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.stl':
        v, f = load_stl(path)
    elif ext == '.dae':
        v, f = load_dae(path)
    else:
        return np.zeros((0, 3))
    return _sample_triangles(v, f, density)


def sample_box(size, density):
    """Uniform samples over the six faces of an axis-aligned box."""
    sx, sy, sz = np.asarray(size, dtype=float)
    faces = [
        ((sy, sz), lambda a, b: np.column_stack([np.full(len(a), sx / 2), a, b])),
        ((sy, sz), lambda a, b: np.column_stack([np.full(len(a), -sx / 2), a, b])),
        ((sx, sz), lambda a, b: np.column_stack([a, np.full(len(a), sy / 2), b])),
        ((sx, sz), lambda a, b: np.column_stack([a, np.full(len(a), -sy / 2), b])),
        ((sx, sy), lambda a, b: np.column_stack([a, b, np.full(len(a), sz / 2)])),
        ((sx, sy), lambda a, b: np.column_stack([a, b, np.full(len(a), -sz / 2)])),
    ]
    rng = np.random.default_rng(1)
    out = []
    for (da, db), place in faces:
        n = max(1, int(da * db * density))
        a = (rng.random(n) - 0.5) * da
        b = (rng.random(n) - 0.5) * db
        out.append(place(a, b))
    return np.vstack(out)


# ---------------------------------------------------------------------------
# nearest neighbour
# ---------------------------------------------------------------------------

class VoxelIndex(object):
    """Grid-hashed exact nearest neighbour, up to a cutoff radius."""

    def __init__(self, points, cell=0.5):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.cell = float(cell)
        self.origin = self.points.min(axis=0) if len(self.points) else np.zeros(3)
        keys = self._keys(self.points)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        self._sorted = order
        sk = keys[order]
        # Start index of each distinct cell in the sorted order.
        diff = np.any(np.diff(sk, axis=0) != 0, axis=1)
        starts = np.concatenate([[0], np.where(diff)[0] + 1])
        ends = np.concatenate([starts[1:], [len(sk)]])
        self._cells = {tuple(sk[s]): (s, e) for s, e in zip(starts, ends)}

    def _keys(self, pts):
        return np.floor((pts - self.origin) / self.cell).astype(np.int64)

    def _candidates(self, key, ring):
        """Reference points in the cells within Chebyshev distance `ring`."""
        chunks = []
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                for dz in range(-ring, ring + 1):
                    cell = self._cells.get((key[0] + dx, key[1] + dy,
                                            key[2] + dz))
                    if cell is not None:
                        s, e = cell
                        chunks.append(self._sorted[s:e])
        if not chunks:
            return None
        return self.points[np.concatenate(chunks)]

    def query(self, queries, max_radius=5.0):
        """Distance to the nearest reference point, per query.

        Queries are grouped by cell so that each group is answered with one
        vectorised distance computation against the candidate points of its
        neighbourhood - the difference between minutes and hours on a
        million-point cloud.

        Exactness: after searching every cell within Chebyshev ring r of the
        query's own cell, the searched region extends at least r*cell from the
        query in every direction, so a best distance of <= r*cell cannot be
        beaten by anything outside.  Queries that fail that test have their
        ring widened; only those still unresolved at max_radius are given up
        on.

        Returns (distances, found).  `found` is False where no reference point
        lies within max_radius, and those distances are NaN - the caller has to
        decide what to do with them rather than having a cutoff value silently
        averaged into the result.
        """
        queries = np.ascontiguousarray(queries, dtype=np.float64)
        n = len(queries)
        dists = np.full(n, np.nan)
        found = np.zeros(n, dtype=bool)
        if len(self.points) == 0 or n == 0:
            return dists, found

        qk = self._keys(queries)
        max_ring = max(1, int(np.ceil(max_radius / self.cell)))

        groups = {}
        for i in range(n):
            groups.setdefault(tuple(qk[i]), []).append(i)

        for key, idx in groups.items():
            idx = np.asarray(idx)
            pending = idx
            ring = 1
            while len(pending) and ring <= max_ring:
                cand = self._candidates(key, ring)
                if cand is None or len(cand) == 0:
                    ring += 1
                    continue
                d = np.linalg.norm(
                    queries[pending][:, None, :] - cand[None, :, :], axis=2)
                best = d.min(axis=1)
                # Exact where the winner is inside the guaranteed radius.
                ok = best <= ring * self.cell
                dists[pending[ok]] = best[ok]
                found[pending[ok]] = True
                # Keep the current best for the rest; it may still be the
                # answer once the ring is wide enough to prove it.
                rest = ~ok
                if rest.any():
                    dists[pending[rest]] = best[rest]
                pending = pending[rest]
                ring += 1

            # Anything still pending has been measured but never proven; accept
            # it if it is inside the cutoff, otherwise mark it unmatched.
            if len(pending):
                d = dists[pending]
                inside = np.isfinite(d) & (d <= max_radius)
                found[pending[inside]] = True
                dists[pending[~inside]] = np.nan

        beyond = np.isfinite(dists) & (dists > max_radius)
        dists[beyond] = np.nan
        found[beyond] = False
        return dists, found
