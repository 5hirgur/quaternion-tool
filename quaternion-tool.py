
import sys
import numpy as np
from PySide6 import QtWidgets
import pyqtgraph.opengl as gl
from PySide6.QtCore import Qt


# ----------------------------
# Geometry: minimal OBJ loader
# ----------------------------
def load_obj_mesh(path):
    """
    Very small OBJ loader: supports only 'v' and 'f' (triangles or polygons).
    Returns vertices (N,3) and faces (M,3) triangulated.
    """
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    idx.append(int(p.split("/")[0]) - 1)
                # triangulate fan
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise ValueError("OBJ had no vertices or faces (or unsupported format).")
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


# ----------------------------
# Quaternion math (wxyz)
# ----------------------------
def parse_floats(text, n):
    t = text.replace(",", " ").strip().split()
    if len(t) != n:
        raise ValueError(f"Need {n} numbers.")
    return [float(v) for v in t]


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # identity in wxyz
    return q / n


def quat_multiply_wxyz(q2, q1):
    """
    Hamilton product: q_total = q2 ⊗ q1
    Interpreted as: apply rotation q1 first, then q2 (active rotation convention).
    Quaternions are in wxyz.
    """
    w2, x2, y2, z2 = q2
    w1, x1, y1, z1 = q1
    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1
    return np.array([w, x, y, z], dtype=float)


def quat_from_axis_angle_wxyz(axis_xyz, angle_rad):
    """
    axis_xyz: (3,) unit or non-unit axis
    angle_rad: float
    returns wxyz unit quaternion
    """
    axis = np.asarray(axis_xyz, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    half = 0.5 * angle_rad
    s = np.sin(half)
    w = np.cos(half)
    x, y, z = axis * s
    return normalize_quat_wxyz([w, x, y, z])


def quat_from_euler_xyz_intrinsic_wxyz(x_rad, y_rad, z_rad):
    """
    Euler XYZ intrinsic (local/body): rotate about local X, then local Y, then local Z.

    Using the composition rule:
      If you apply X then Y then Z, the combined quaternion is:
        q_delta = q_z ⊗ q_y ⊗ q_x
      because q2 ⊗ q1 means "q1 then q2".

    Returns wxyz.
    """
    qx = quat_from_axis_angle_wxyz([1, 0, 0], x_rad)
    qy = quat_from_axis_angle_wxyz([0, 1, 0], y_rad)
    qz = quat_from_axis_angle_wxyz([0, 0, 1], z_rad)
    return quat_multiply_wxyz(qz, quat_multiply_wxyz(qy, qx))


def quat_to_rotmat_from_wxyz(q_wxyz):
    """
    Convert wxyz quaternion to a 3x3 rotation matrix.
    Internally uses the standard xyzw formula by reordering.
    """
    q = normalize_quat_wxyz(q_wxyz)
    w, x, y, z = q
    # Standard rotation matrix from wxyz directly:
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=float)


def make_qmatrix4x4_from_rot3(R):
    """
    Build a Qt-friendly 4x4 transform matrix (QMatrix4x4) from a 3x3 rotation.
    """
    from PySide6.QtGui import QMatrix4x4
    M = np.eye(4, dtype=float)
    M[:3, :3] = R
    # QMatrix4x4 constructor accepts 16 scalars in row-major order
    return QMatrix4x4(*M.reshape(-1).tolist())


# ----------------------------
# App
# ----------------------------
class QuatViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quaternion Chaining Viewer (w x y z) ")

        # State
        self.q_current = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # wxyz
        self.ghosts = []
        self.trail_limit = 25

        # Layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Controls (left)
        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)

        # Quaternion absolute
        cl.addWidget(QtWidgets.QLabel("Absolute quaternion (w x y z):"))
        self.q_edit = QtWidgets.QLineEdit("1 0 0 0")
        cl.addWidget(self.q_edit)

        self.btn_set = QtWidgets.QPushButton("Set orientation")
        self.btn_set.clicked.connect(self.on_set_orientation)
        cl.addWidget(self.btn_set)

        cl.addSpacing(10)

        # Euler delta
        cl.addWidget(QtWidgets.QLabel("Delta Euler angles (intrinsic/local XYZ):"))
        row = QtWidgets.QHBoxLayout()
        self.ex = QtWidgets.QLineEdit("0")
        self.ey = QtWidgets.QLineEdit("0")
        self.ez = QtWidgets.QLineEdit("0")
        self.ex.setPlaceholderText("X")
        self.ey.setPlaceholderText("Y")
        self.ez.setPlaceholderText("Z")
        row.addWidget(self.ex)
        row.addWidget(self.ey)
        row.addWidget(self.ez)
        cl.addLayout(row)

        self.units = QtWidgets.QComboBox()
        self.units.addItems(["Degrees", "Radians"])
        self.units.setCurrentText("Degrees")
        cl.addWidget(self.units)

        self.btn_add = QtWidgets.QPushButton("Add rotation (chain)")
        self.btn_add.clicked.connect(self.on_add_rotation)
        cl.addWidget(self.btn_add)

        cl.addSpacing(10)

        # Trail controls
        trail_row = QtWidgets.QHBoxLayout()
        self.btn_clear_trail = QtWidgets.QPushButton("Clear trail")
        self.btn_clear_trail.clicked.connect(self.on_clear_trail)
        trail_row.addWidget(self.btn_clear_trail)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        trail_row.addWidget(self.btn_reset)
        cl.addLayout(trail_row)

        cl.addSpacing(10)

        # Output text
        cl.addWidget(QtWidgets.QLabel("Current (normalized) quaternion w x y z:"))
        self.out_q = QtWidgets.QLabel("1 0 0 0")
        self.out_q.setTextInteractionFlags(
        self.out_q.textInteractionFlags() | Qt.TextSelectableByMouse
        )
        cl.addWidget(self.out_q)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        cl.addWidget(self.status)

        cl.addStretch(1)

        # 3D view (right)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=8, elevation=20, azimuth=35)

        root.addWidget(controls, 0)
        root.addWidget(self.view, 1)

        # Scene helpers
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.view.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.view.addItem(axis)

        # Load mesh data once and reuse for main + ghosts
        self.meshdata = self.load_meshdata("penguin.obj")

        # Main mesh
        self.main_item = gl.GLMeshItem(
            meshdata=self.meshdata,
            smooth=True,
            shader="shaded",
            drawEdges=False,
            color=(0.85, 0.85, 0.95, 1.0),
        )
        self.view.addItem(self.main_item)

        # Apply identity
        self.apply_quaternion_to_item(self.main_item, self.q_current)
        self.refresh_outputs()

    def load_meshdata(self, path):
        try:
            verts, faces = load_obj_mesh(path)

            # Center + scale for nicer viewing
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
            center = (vmin + vmax) * 0.5
            extent = np.linalg.norm(vmax - vmin)
            scale = 3.0 / extent if extent > 0 else 1.0
            verts = (verts - center) * scale

            return gl.MeshData(vertexes=verts, faces=faces)
        except Exception as e:
            self.status.setText(
                f"Could not load '{path}' ({e}). Showing placeholder sphere. "
                f"Put a penguin OBJ named 'penguin.obj' next to the script."
            )
            return gl.MeshData.sphere(rows=16, cols=32, radius=1.0)

    def apply_quaternion_to_item(self, item, q_wxyz):
        R = quat_to_rotmat_from_wxyz(q_wxyz)
        item.resetTransform()
        item.setTransform(make_qmatrix4x4_from_rot3(R))

    def refresh_outputs(self):
        self.q_current = normalize_quat_wxyz(self.q_current)
        w, x, y, z = self.q_current
        self.out_q.setText(f"{w:.6f}  {x:.6f}  {y:.6f}  {z:.6f}")

    # ----------------------------
    # UI handlers
    # ----------------------------
    def on_set_orientation(self):
        try:
            q_in = parse_floats(self.q_edit.text(), 4)  # w x y z
            q = normalize_quat_wxyz(q_in)

            # Set absolute orientation (no trail by default)
            self.q_current = q
            self.apply_quaternion_to_item(self.main_item, self.q_current)
            self.refresh_outputs()
            self.status.setText("Set absolute orientation.")
        except Exception as e:
            self.status.setText(f"Error: {e}")

    def on_add_rotation(self):
        try:
            # Create ghost of current pose
            self.add_ghost_pose(self.q_current)

            # Parse Euler delta
            x = float(self.ex.text().strip() or "0")
            y = float(self.ey.text().strip() or "0")
            z = float(self.ez.text().strip() or "0")

            if self.units.currentText() == "Degrees":
                x = np.deg2rad(x)
                y = np.deg2rad(y)
                z = np.deg2rad(z)

            # Euler -> delta quaternion (intrinsic/local XYZ)
            q_delta = quat_from_euler_xyz_intrinsic_wxyz(x, y, z)

            # Chain in LOCAL/BODY frame:
            # q_new = q_current ⊗ q_delta
            self.q_current = quat_multiply_wxyz(self.q_current, q_delta)
            self.q_current = normalize_quat_wxyz(self.q_current)

            # Apply
            self.apply_quaternion_to_item(self.main_item, self.q_current)
            self.refresh_outputs()
            self.status.setText("Added intrinsic/local XYZ delta rotation (chained).")
        except Exception as e:
            self.status.setText(f"Error: {e}")

    def add_ghost_pose(self, q_wxyz):
        ghost = gl.GLMeshItem(
            meshdata=self.meshdata,
            smooth=True,
            shader="shaded",
            drawEdges=False,
            color=(1.0, 0.45, 0.45, 0.28),  # light red, semi-transparent
        )
        self.apply_quaternion_to_item(ghost, q_wxyz)
        self.view.addItem(ghost)
        self.ghosts.append(ghost)

        # Cap trail length
        while len(self.ghosts) > self.trail_limit:
            old = self.ghosts.pop(0)
            self.view.removeItem(old)

    def on_clear_trail(self):
        for g in self.ghosts:
            self.view.removeItem(g)
        self.ghosts.clear()
        self.status.setText("Trail cleared.")

    def on_reset(self):
        self.on_clear_trail()
        self.q_current = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # identity
        self.apply_quaternion_to_item(self.main_item, self.q_current)
        self.refresh_outputs()
        self.status.setText("Reset to identity orientation.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = QuatViewer()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())
