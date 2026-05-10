"""
Интерактивная симуляция магнитного поля соленоида (осесимметричная задача).
Поле каждого витка считается аналитически через полные эллиптические интегралы,
поле соленоида — суперпозиция N витков, равномерно распределённых вдоль оси.

Запуск:  python coil_simulation.py
Зависимости:  PySide6, numpy, matplotlib
"""

import sys
import numpy as np


def _ellip_km_em(m):
    """Полные эллиптические интегралы K(m) и E(m) через AGM.
    m = k^2 в диапазоне [0, 1). Векторизовано по numpy."""
    m = np.asarray(m, dtype=float)
    a = np.ones_like(m)
    b = np.sqrt(np.maximum(1.0 - m, 0.0))
    c = np.sqrt(np.maximum(m, 0.0))
    sum_c2 = 0.5 * c * c
    factor = 1.0
    for _ in range(20):
        a_new = 0.5 * (a + b)
        b_new = np.sqrt(a * b)
        c = 0.5 * (a - b)
        a, b = a_new, b_new
        factor *= 2.0
        sum_c2 = sum_c2 + factor * 0.5 * c * c
        if np.max(np.abs(c)) < 1e-15:
            break
    K = np.pi / (2.0 * a)
    E = K * (1.0 - sum_c2)
    return K, E


def ellipk(m):
    return _ellip_km_em(m)[0]


def ellipe(m):
    return _ellip_km_em(m)[1]

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout,
    QCheckBox, QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

MU0 = 4.0 * np.pi * 1e-7


def loop_field(I, a, z0, rho, z):
    """Поле кругового витка радиуса a с током I, центр на оси в точке z=z0.
    rho, z — массивы координат точек наблюдения (в плоскости r-z).
    Возвращает (B_rho, B_z) в Тл."""
    dz = z - z0
    rho_abs = np.abs(rho)
    rho_safe = np.where(rho_abs < 1e-12, 1e-12, rho_abs)

    alpha2 = a * a + rho_abs * rho_abs + dz * dz - 2.0 * a * rho_abs
    beta2 = a * a + rho_abs * rho_abs + dz * dz + 2.0 * a * rho_abs
    alpha2 = np.maximum(alpha2, 1e-20)
    beta = np.sqrt(beta2)

    k2 = 1.0 - alpha2 / beta2
    k2 = np.clip(k2, 0.0, 1.0 - 1e-12)

    K = ellipk(k2)
    E = ellipe(k2)
    C = MU0 * I / np.pi

    Bz = C / (2.0 * alpha2 * beta) * (
        (a * a - rho_abs * rho_abs - dz * dz) * E + alpha2 * K
    )
    Br_mag = C * dz / (2.0 * alpha2 * beta * rho_safe) * (
        (a * a + rho_abs * rho_abs + dz * dz) * E - alpha2 * K
    )
    Br_mag = np.where(rho_abs < 1e-12, 0.0, Br_mag)
    Br = np.sign(rho) * Br_mag
    return Br, Bz


def solenoid_field(I, a, L, N, rho, z):
    """Суперпозиция N равномерно распределённых витков от -L/2 до +L/2."""
    Br_total = np.zeros_like(rho, dtype=float)
    Bz_total = np.zeros_like(rho, dtype=float)
    if N <= 1:
        z_positions = np.array([0.0])
    else:
        z_positions = np.linspace(-L / 2.0, L / 2.0, N)
    for z0 in z_positions:
        Br, Bz = loop_field(I, a, z0, rho, z)
        Br_total += Br
        Bz_total += Bz
    return Br_total, Bz_total, z_positions


class CoilSimWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Магнитное поле соленоида — PySide6")
        self.resize(1200, 750)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_plot(), 1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self.recompute)

        self.recompute()

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Параметры катушки")
        form = QFormLayout(box)

        self.spin_I = QDoubleSpinBox()
        self.spin_I.setRange(-1000.0, 1000.0)
        self.spin_I.setDecimals(2)
        self.spin_I.setSingleStep(0.5)
        self.spin_I.setSuffix(" А")
        self.spin_I.setValue(5.0)

        self.spin_R = QDoubleSpinBox()
        self.spin_R.setRange(0.5, 500.0)
        self.spin_R.setDecimals(1)
        self.spin_R.setSingleStep(1.0)
        self.spin_R.setSuffix(" мм")
        self.spin_R.setValue(25.0)

        self.spin_L = QDoubleSpinBox()
        self.spin_L.setRange(0.0, 2000.0)
        self.spin_L.setDecimals(1)
        self.spin_L.setSingleStep(2.0)
        self.spin_L.setSuffix(" мм")
        self.spin_L.setValue(80.0)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(1, 500)
        self.spin_N.setValue(40)

        self.combo_view = QComboBox()
        self.combo_view.addItems(["Линии поля + |B|", "Только линии", "Только |B|", "Векторы"])

        self.chk_log = QCheckBox("Лог. шкала |B|")
        self.chk_log.setChecked(True)

        self.chk_show_coil = QCheckBox("Показать витки")
        self.chk_show_coil.setChecked(True)

        form.addRow("Ток I:", self.spin_I)
        form.addRow("Радиус R:", self.spin_R)
        form.addRow("Длина L:", self.spin_L)
        form.addRow("Витков N:", self.spin_N)
        form.addRow("Режим:", self.combo_view)
        form.addRow(self.chk_log)
        form.addRow(self.chk_show_coil)

        self.lbl_b_center = QLabel("B(0,0) = —")
        self.lbl_b_ideal = QLabel("B_ид = —")
        self.lbl_flux = QLabel("Φ ≈ —")
        for lbl in (self.lbl_b_center, self.lbl_b_ideal, self.lbl_flux):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        info = QGroupBox("Расчёт в центре")
        info_l = QFormLayout(info)
        info_l.addRow(self.lbl_b_center)
        info_l.addRow(self.lbl_b_ideal)
        info_l.addRow(self.lbl_flux)

        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.addWidget(box)
        v.addWidget(info)
        v.addStretch(1)
        wrap.setFixedWidth(290)

        for w in (self.spin_I, self.spin_R, self.spin_L):
            w.valueChanged.connect(self._debounced)
        self.spin_N.valueChanged.connect(self._debounced)
        self.combo_view.currentIndexChanged.connect(self._debounced)
        self.chk_log.stateChanged.connect(self._debounced)
        self.chk_show_coil.stateChanged.connect(self._debounced)

        return wrap

    def _build_plot(self) -> QWidget:
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)

        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas)
        return wrap

    def _debounced(self, *_):
        self._debounce.start()

    def recompute(self):
        I = float(self.spin_I.value())
        R = float(self.spin_R.value()) * 1e-3
        L = float(self.spin_L.value()) * 1e-3
        N = int(self.spin_N.value())
        mode = self.combo_view.currentIndex()
        use_log = self.chk_log.isChecked()
        show_coil = self.chk_show_coil.isChecked()

        extent_z = max(L * 1.6, R * 4.0, 0.04)
        extent_r = max(R * 3.5, 0.04)

        nz, nr = 220, 180
        z_lin = np.linspace(-extent_z, extent_z, nz)
        r_lin = np.linspace(-extent_r, extent_r, nr)
        Z, R_grid = np.meshgrid(z_lin, r_lin)

        Br, Bz, z_positions = solenoid_field(I, R, L, N, R_grid, Z)
        Bmag = np.sqrt(Br * Br + Bz * Bz) + 1e-30

        ax = self.ax
        ax.clear()

        if mode in (0, 2):
            if use_log:
                vmin = max(np.percentile(Bmag, 5), 1e-9)
                vmax = max(np.percentile(Bmag, 99.5), vmin * 10)
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = None
            pcm = ax.pcolormesh(
                Z * 1e3, R_grid * 1e3, Bmag,
                shading="auto", cmap="inferno", norm=norm,
            )
            if hasattr(self, "_cbar") and self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
            self._cbar = self.fig.colorbar(pcm, ax=ax, pad=0.02)
            self._cbar.set_label("|B|, Тл")
        else:
            self._cbar = None

        if mode in (0, 1):
            try:
                ax.streamplot(
                    Z * 1e3, R_grid * 1e3, Bz, Br,
                    color="white" if mode == 0 else "tab:blue",
                    density=1.4, linewidth=0.8, arrowsize=0.9,
                )
            except Exception:
                pass

        if mode == 3:
            step = 10
            ax.quiver(
                Z[::step, ::step] * 1e3, R_grid[::step, ::step] * 1e3,
                Bz[::step, ::step], Br[::step, ::step],
                Bmag[::step, ::step], cmap="viridis",
                pivot="mid", scale_units="xy",
            )

        if show_coil:
            for z0 in z_positions:
                ax.plot(z0 * 1e3, R * 1e3, marker="o",
                        color="tab:cyan", markersize=4, markeredgecolor="black")
                ax.plot(z0 * 1e3, -R * 1e3, marker="x",
                        color="tab:cyan", markersize=5, markeredgewidth=1.5)
            ax.plot([-L / 2 * 1e3, L / 2 * 1e3], [R * 1e3, R * 1e3],
                    color="tab:cyan", lw=0.8, alpha=0.5)
            ax.plot([-L / 2 * 1e3, L / 2 * 1e3], [-R * 1e3, -R * 1e3],
                    color="tab:cyan", lw=0.8, alpha=0.5)

        ax.set_xlabel("z, мм")
        ax.set_ylabel("r, мм")
        ax.set_title(
            f"Соленоид: I = {I:g} А, R = {R*1e3:g} мм, L = {L*1e3:g} мм, N = {N}"
        )
        ax.set_aspect("equal")
        ax.set_xlim(-extent_z * 1e3, extent_z * 1e3)
        ax.set_ylim(-extent_r * 1e3, extent_r * 1e3)

        Br_c, Bz_c, _ = solenoid_field(I, R, L, N, np.array([0.0]), np.array([0.0]))
        B_center = float(np.hypot(Br_c[0], Bz_c[0]))
        if L > 0 and N > 0:
            n_per_m = N / L
            B_ideal = MU0 * n_per_m * I
        else:
            B_ideal = 0.0
        flux = float(B_center) * np.pi * R * R

        self.lbl_b_center.setText(f"|B|(0,0) = {B_center*1e3:.4g} мТл")
        self.lbl_b_ideal.setText(f"B_ид (∞-соленоид) = {B_ideal*1e3:.4g} мТл")
        self.lbl_flux.setText(f"Φ ≈ {flux*1e6:.4g} мкВб")

        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = CoilSimWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
