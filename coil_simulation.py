"""
Динамичная симуляция магнитного поля катушки.
PySide6 + QPainter, тёмная тема. Без matplotlib и scipy.

Что показано:
  - Сечение соленоида (вид сбоку); каждый виток — точка-маркер с символом
    направления тока (× — в плоскость экрана, • — из неё) по правилу правой руки.
  - Замкнутые «кольца» — линии магнитного поля в меридиональной плоскости
    (внутри катушки идут вдоль оси, снаружи замыкаются через торцы).
  - Светящиеся частицы текут вдоль линий: направление потока определяется
    знаком тока, скорость и плотность — его величиной.

Запуск:  python coil_simulation.py
"""

import sys
import math
import random

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QElapsedTimer
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPainterPath,
    QRadialGradient, QLinearGradient, QFont,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QPushButton,
)

MU0 = 4.0 * math.pi * 1e-7


def b_on_axis(I, R, L, N, z=0.0):
    """Аналитическое поле на оси конечного соленоида (Тл)."""
    if L <= 0 or N <= 0:
        return 0.0
    n = N / L
    a = (z + L / 2.0) / math.sqrt((z + L / 2.0) ** 2 + R * R)
    b = (z - L / 2.0) / math.sqrt((z - L / 2.0) ** 2 + R * R)
    return 0.5 * MU0 * n * I * (a - b)


def b_ideal(I, R, L, N):
    if L <= 0:
        return 0.0
    return MU0 * (N / L) * I


def field_line_points(r_in, r_out, L, cap_extra, n=240):
    """Замкнутая линия поля в верхней полуплоскости в форме «стадиона»:
       нижняя сторона — внутри катушки (r=r_in), верхняя — снаружи (r=r_out),
       торцы — полу-эллиптические шапки, вылезающие за катушку на cap_extra.

       Параметризация t∈[0,1) такова, что рост t = направление потока для I>0:
       сверху наружу поток в −z, снизу внутри — в +z.
    """
    pts = []
    L2 = L / 2.0
    r_mid = (r_in + r_out) / 2.0
    r_amp = (r_out - r_in) / 2.0
    for i in range(n):
        t = i / n
        if t < 0.4:                                    # верх (снаружи), z: +L/2 → −L/2
            u = t / 0.4
            z = L2 - u * L
            r = r_out
        elif t < 0.5:                                  # левая шапка (вылезает в −z)
            u = (t - 0.4) / 0.1
            theta = math.pi / 2.0 - u * math.pi
            z = -L2 - cap_extra * math.cos(theta)
            r = r_mid + r_amp * math.sin(theta)
        elif t < 0.9:                                  # низ (внутри), z: −L/2 → +L/2
            u = (t - 0.5) / 0.4
            z = -L2 + u * L
            r = r_in
        else:                                          # правая шапка (вылезает в +z)
            u = (t - 0.9) / 0.1
            theta = -math.pi / 2.0 + u * math.pi
            z = L2 + cap_extra * math.cos(theta)
            r = r_mid + r_amp * math.sin(theta)
        pts.append((z, r))
    return pts


class CoilFieldWidget(QWidget):
    BG_TOP = QColor(20, 25, 35)
    BG_BOT = QColor(8, 12, 18)
    AXIS = QColor(255, 255, 255, 22)
    COIL_BORDER = QColor(110, 175, 220)
    COIL_FILL_OFF = QColor(110, 175, 220, 25)
    COIL_FILL_ON = QColor(110, 175, 220, 60)
    LINE_COLOR = QColor(80, 140, 200)
    PARTICLE_GLOW_FWD = QColor(110, 220, 255)
    PARTICLE_GLOW_REV = QColor(255, 130, 200)
    WIRE_RING = QColor(180, 200, 220, 200)
    WIRE_FG = QColor(230, 240, 250)
    WIRE_BG_OFF = QColor(45, 50, 60)
    WIRE_BG_ON = QColor(40, 75, 105)
    TEXT = QColor(180, 200, 220)
    TEXT_DIM = QColor(120, 140, 160)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 520)
        self.setAutoFillBackground(False)

        self.I = 5.0
        self.R_coil = 0.04
        self.L_coil = 0.10
        self.N_turns = 30
        self.I_ref = 20.0

        self.field_lines = []
        self._rebuild_field_lines()
        self.particles = []

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._last_t = 0.0
        self._spawn_acc = 0.0
        self._timer.start()

    # ---- public ----
    def set_current(self, I):
        self.I = float(I)
        self.update()

    def set_radius_mm(self, v):
        self.R_coil = max(float(v), 1.0) * 1e-3
        self._rebuild_field_lines()
        self.update()

    def set_length_mm(self, v):
        self.L_coil = max(float(v), 0.5) * 1e-3
        self._rebuild_field_lines()
        self.update()

    def set_turns(self, v):
        self.N_turns = max(1, int(v))
        self.update()

    # ---- internals ----
    def _rebuild_field_lines(self):
        self.field_lines = []
        scales = [0.20, 0.40, 0.62, 0.85, 1.10, 1.40, 1.75]
        for i, s in enumerate(scales):
            r_in = max(0.10 * self.R_coil, self.R_coil * (0.92 - 0.45 * s))
            r_out = self.R_coil * (1.0 + 1.2 * s)
            cap_extra = self.R_coil * (0.30 + 0.40 * s)
            pts = field_line_points(r_in, r_out, self.L_coil, cap_extra, n=240)
            self.field_lines.append({
                "scale": s, "r_in": r_in, "r_out": r_out,
                "cap_extra": cap_extra, "points": pts,
                "threshold": 0.06 + 0.13 * i,
            })
        self.particles = [p for p in self.particles if p["line"] < len(self.field_lines)]

    def _tick(self):
        t_now = self._clock.elapsed() / 1000.0
        dt = max(1e-6, min(0.1, t_now - self._last_t))
        self._last_t = t_now

        I_abs = abs(self.I)
        I_norm = min(1.0, I_abs / self.I_ref)
        sign = 1.0 if self.I >= 0 else -1.0
        speed = (0.04 + 0.55 * I_norm) if I_abs > 0.01 else 0.0
        spawn_rate = 0.0 if I_abs < 0.05 else (4.0 + 80.0 * I_norm)

        kept = []
        for p in self.particles:
            p["t"] = (p["t"] + speed * dt * sign) % 1.0
            p["age"] += dt
            if p["age"] < p["max_life"]:
                kept.append(p)
        self.particles = kept

        if spawn_rate > 0 and self.field_lines:
            self._spawn_acc += spawn_rate * dt
            n_spawn = int(self._spawn_acc)
            self._spawn_acc -= n_spawn
            for _ in range(min(n_spawn, 25)):
                eligible = [i for i, ln in enumerate(self.field_lines)
                            if I_norm >= ln["threshold"] * 0.45]
                if not eligible:
                    eligible = [0]
                line_idx = random.choice(eligible)
                side = random.choice([-1, 1])
                self.particles.append({
                    "line": line_idx, "side": side,
                    "t": random.random(),
                    "age": 0.0,
                    "max_life": 1.6 + 2.6 * random.random(),
                })

        if len(self.particles) > 700:
            self.particles = self.particles[-700:]

        self.update()

    # ---- painting ----
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        bg = QLinearGradient(0, 0, 0, rect.height())
        bg.setColorAt(0, self.BG_TOP)
        bg.setColorAt(1, self.BG_BOT)
        p.fillRect(rect, bg)

        if not self.field_lines:
            p.end()
            return

        max_R = max(ln["r_out"] for ln in self.field_lines)
        max_cap = max(ln["cap_extra"] for ln in self.field_lines)
        margin = 1.10
        world_hw = (self.L_coil / 2.0 + max_cap) * margin
        world_hh = max_R * margin
        sx = (rect.width() - 50) / (2.0 * world_hw)
        sy = (rect.height() - 50) / (2.0 * world_hh)
        scale = min(sx, sy)
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0

        def w2s(z, r):
            return QPointF(cx + z * scale, cy - r * scale)

        # axis
        p.setPen(QPen(self.AXIS, 1, Qt.DashLine))
        p.drawLine(w2s(-world_hw, 0), w2s(world_hw, 0))

        I_abs = abs(self.I)
        I_norm = min(1.0, I_abs / self.I_ref)

        # field-line backbones
        for ln in self.field_lines:
            if I_abs < 0.05:
                visibility = 0.5
            else:
                visibility = max(0.0, min(1.0, (I_norm - ln["threshold"] * 0.45) / 0.30 + 0.15))
            if visibility <= 0.0:
                continue
            for side in (+1, -1):
                path = QPainterPath()
                for i, (zw, rw) in enumerate(ln["points"]):
                    pt = w2s(zw, side * rw)
                    if i == 0:
                        path.moveTo(pt)
                    else:
                        path.lineTo(pt)
                path.closeSubpath()
                alpha = int(20 + 80 * visibility)
                p.setPen(QPen(QColor(self.LINE_COLOR.red(), self.LINE_COLOR.green(),
                                     self.LINE_COLOR.blue(), alpha), 1.2))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)

        # particles
        glow = self.PARTICLE_GLOW_REV if self.I < 0 else self.PARTICLE_GLOW_FWD
        for prt in self.particles:
            ln = self.field_lines[prt["line"]]
            pts = ln["points"]
            n_pts = len(pts)
            i_f = prt["t"] * n_pts
            i_a = int(i_f) % n_pts
            i_b = (i_a + 1) % n_pts
            f = i_f - int(i_f)
            za, ra = pts[i_a]
            zb, rb = pts[i_b]
            zw = za + (zb - za) * f
            rw = ra + (rb - ra) * f
            pt = w2s(zw, prt["side"] * rw)
            life_frac = max(0.0, 1.0 - prt["age"] / prt["max_life"])
            fade_in = min(1.0, prt["age"] / 0.18)
            self._draw_particle(p, pt, life_frac * fade_in, glow)

        # coil cross-section
        l2 = self.L_coil / 2.0
        tl = w2s(-l2, +self.R_coil)
        br = w2s(+l2, -self.R_coil)
        coil_rect = QRectF(tl, br)
        body = QLinearGradient(coil_rect.topLeft(), coil_rect.bottomLeft())
        fill = self.COIL_FILL_ON if I_abs > 0.05 else self.COIL_FILL_OFF
        body.setColorAt(0, fill)
        body.setColorAt(0.5, QColor(fill.red(), fill.green(), fill.blue(),
                                    max(0, fill.alpha() - 30)))
        body.setColorAt(1, fill)
        p.setBrush(body)
        p.setPen(QPen(self.COIL_BORDER, 1.6))
        p.drawRoundedRect(coil_rect, 4, 4)

        # wires (turns)
        N = self.N_turns
        if N > 0:
            active = I_abs > 0.05
            top_into = self.I >= 0
            if N == 1:
                zs = [0.0]
            else:
                zs = [-l2 + i * self.L_coil / (N - 1) for i in range(N)]
            for z0 in zs:
                self._draw_wire(p, w2s(z0, +self.R_coil), top_into, active)
                self._draw_wire(p, w2s(z0, -self.R_coil), not top_into, active)

        # info text
        p.setFont(QFont("Segoe UI", 9))
        Bc = b_on_axis(self.I, self.R_coil, self.L_coil, self.N_turns, 0.0)
        Bi = b_ideal(self.I, self.R_coil, self.L_coil, self.N_turns)
        info1 = (f"I = {self.I:+6.2f} А    "
                 f"R = {self.R_coil*1000:.1f} мм    "
                 f"L = {self.L_coil*1000:.1f} мм    "
                 f"N = {self.N_turns}")
        info2 = (f"|B|(0,0) = {Bc*1e3:+.4g} мТл    "
                 f"B_ид = µ₀nI = {Bi*1e3:+.4g} мТл    "
                 f"частиц = {len(self.particles)}")
        p.setPen(self.TEXT)
        p.drawText(15, 22, info1)
        p.setPen(self.TEXT_DIM)
        p.drawText(15, rect.height() - 12, info2)

        p.end()

    def _draw_particle(self, p, pt, alpha, glow):
        if alpha <= 0:
            return
        a_halo = int(150 * alpha)
        a_core = int(255 * alpha)
        grad = QRadialGradient(pt, 11)
        grad.setColorAt(0.0, QColor(255, 255, 240, a_core))
        grad.setColorAt(0.4, QColor(glow.red(), glow.green(), glow.blue(), a_halo))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(pt, 11, 11)
        p.setBrush(QColor(255, 255, 250, a_core))
        p.drawEllipse(pt, 1.8, 1.8)

    def _draw_wire(self, p, pt, into, active):
        size = 7.0
        bg = self.WIRE_BG_ON if active else self.WIRE_BG_OFF
        p.setPen(QPen(self.WIRE_RING, 1.4))
        p.setBrush(bg)
        p.drawEllipse(pt, size, size)
        if active:
            p.setPen(QPen(self.WIRE_FG, 1.6))
            if into:
                d = size * 0.55
                p.drawLine(QPointF(pt.x() - d, pt.y() - d),
                           QPointF(pt.x() + d, pt.y() + d))
                p.drawLine(QPointF(pt.x() - d, pt.y() + d),
                           QPointF(pt.x() + d, pt.y() - d))
            else:
                p.setBrush(self.WIRE_FG)
                p.drawEllipse(pt, 2.2, 2.2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Магнитное поле катушки — PySide6")
        self.resize(1100, 700)
        self._apply_dark_style()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.canvas = CoilFieldWidget()
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self.canvas, 1)

    def _apply_dark_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0d1117; color: #c9d1d9; }
            QGroupBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px; padding: 0 6px;
                color: #58a6ff;
            }
            QLabel { color: #c9d1d9; }
            QSlider::groove:horizontal {
                border: 1px solid #30363d;
                height: 6px; background: #21262d; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff; border: 1px solid #58a6ff;
                width: 16px; margin: -6px 0; border-radius: 8px;
            }
            QSlider::handle:horizontal:hover { background: #79b8ff; }
            QSlider::sub-page:horizontal { background: #1f6feb; border-radius: 3px; }
            QSpinBox, QDoubleSpinBox {
                background: #21262d; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 4px;
                padding: 3px 6px; min-width: 80px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus { border-color: #58a6ff; }
            QPushButton {
                background: #21262d; color: #c9d1d9;
                border: 1px solid #30363d; border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover { background: #30363d; border-color: #8b949e; }
            QPushButton:pressed { background: #1f6feb; }
        """)

    def _build_controls(self) -> QWidget:
        wrap = QWidget()
        wrap.setFixedWidth(290)
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        gb = QGroupBox("Параметры")
        f = QFormLayout(gb)
        f.setLabelAlignment(Qt.AlignRight)

        self.sld_I = QSlider(Qt.Horizontal)
        self.sld_I.setRange(-2000, 2000)  # ±20.00 А c шагом 0.01
        self.sld_I.setValue(500)
        self.lbl_I = QLabel("+5.00 А")
        self.lbl_I.setMinimumWidth(70)
        self.sld_I.valueChanged.connect(self._on_I)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.sld_I)
        rl.addWidget(self.lbl_I)
        f.addRow("Ток I:", row)

        self.spn_R = QDoubleSpinBox()
        self.spn_R.setRange(2.0, 200.0)
        self.spn_R.setSuffix(" мм")
        self.spn_R.setSingleStep(1.0)
        self.spn_R.setDecimals(1)
        self.spn_R.setValue(40.0)
        self.spn_R.valueChanged.connect(lambda v: self.canvas.set_radius_mm(v))
        f.addRow("Радиус R:", self.spn_R)

        self.spn_L = QDoubleSpinBox()
        self.spn_L.setRange(2.0, 500.0)
        self.spn_L.setSuffix(" мм")
        self.spn_L.setSingleStep(2.0)
        self.spn_L.setDecimals(1)
        self.spn_L.setValue(100.0)
        self.spn_L.valueChanged.connect(lambda v: self.canvas.set_length_mm(v))
        f.addRow("Длина L:", self.spn_L)

        self.spn_N = QSpinBox()
        self.spn_N.setRange(1, 200)
        self.spn_N.setValue(30)
        self.spn_N.valueChanged.connect(lambda v: self.canvas.set_turns(v))
        f.addRow("Витков N:", self.spn_N)

        v.addWidget(gb)

        btn_off = QPushButton("Выключить ток")
        btn_off.clicked.connect(lambda: self.sld_I.setValue(0))
        btn_rev = QPushButton("Реверс ⇄")
        btn_rev.clicked.connect(lambda: self.sld_I.setValue(-self.sld_I.value()))
        btn_max = QPushButton("Максимум +")
        btn_max.clicked.connect(lambda: self.sld_I.setValue(2000))
        btn_min = QPushButton("Максимум −")
        btn_min.clicked.connect(lambda: self.sld_I.setValue(-2000))
        v.addWidget(btn_off)
        v.addWidget(btn_rev)
        v.addWidget(btn_max)
        v.addWidget(btn_min)

        gb2 = QGroupBox("Подсказка")
        v2 = QVBoxLayout(gb2)
        info = QLabel(
            "× — ток уходит в плоскость экрана\n"
            "•  — ток выходит из плоскости экрана\n\n"
            "Замкнутые контуры — линии магнитного поля.\n"
            "Светящиеся точки текут вдоль линий: направление потока зависит от знака тока, "
            "плотность и скорость — от его величины."
        )
        info.setWordWrap(True)
        v2.addWidget(info)
        v.addWidget(gb2)

        v.addStretch(1)
        return wrap

    def _on_I(self, val):
        I = val / 100.0
        self.lbl_I.setText(f"{I:+.2f} А")
        self.canvas.set_current(I)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
