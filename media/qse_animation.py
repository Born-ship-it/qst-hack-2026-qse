"""
Quantum Subspace Expansion visualisation.

Three sequential stages laid out in three columns:

    1. Krylov basis  |  2. Swap test  |  3. GEVP → E_0
"""

from manim import *
import numpy as np


class QseAnimation(Scene):

    COLS = [-4.6, 0.0, 4.6]
    Y_TITLE = 2.3

    def construct(self):
        title = Text("Quantum Subspace Expansion", font_size=32)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=0.6)
        self.wait(0.15)

        counter = self._counter(0, 3)
        self.add(counter)

        self._stage1_krylov()
        counter = self._advance_counter(counter, 1, 3)

        self._stage2_swap_test()
        counter = self._advance_counter(counter, 2, 3)

        self._stage3_gevp()
        counter = self._advance_counter(counter, 3, 3)

        self.wait(2.0)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage1_krylov(self):
        x = self.COLS[0]
        t = self._stage_title(x, "1. Krylov basis", BLUE_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"|\psi_\ell\rangle = e^{-i\ell \Delta t H}|\psi_0\rangle",
            font_size=22, color=BLUE_C,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.7)

        # Row of Krylov vectors
        circles = VGroup()
        labels = VGroup()
        n = 5
        spacing = 0.7
        for i in range(n):
            c = Circle(
                radius=0.28, color=BLUE_C,
                fill_color=BLUE_D, fill_opacity=0.4,
                stroke_width=2.5,
            ).move_to([x + (i - (n - 1) / 2) * spacing, -0.6, 0])
            circles.add(c)

            lbl = MathTex(
                rf"|\psi_{i}\rangle", font_size=18, color=BLUE_C,
            ).move_to(c.get_center())
            labels.add(lbl)

        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in circles],
                        lag_ratio=0.1),
            run_time=0.8,
        )
        self.play(FadeIn(labels), run_time=0.5)
        self.wait(0.3)

    def _stage2_swap_test(self):
        x = self.COLS[1]
        t = self._stage_title(x, "2. Extended swap test", YELLOW_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"\langle \psi_k | H | \psi_\ell \rangle",
            font_size=22, color=YELLOW_C,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.7)

        # Circuit diagram
        anc = Line([-2, 0.2, 0], [2, 0.2, 0],
                   stroke_width=2.5, color=GREY_B)
        sys = Line([-2, -0.6, 0], [2, -0.6, 0],
                   stroke_width=2.5, color=GREY_B)

        anc_lbl = Text("ancilla", font_size=14, color=GREY_B)
        anc_lbl.next_to(anc, UP, buff=0.08)
        sys_lbl = Text("system", font_size=14, color=GREY_B)
        sys_lbl.next_to(sys, UP, buff=0.08)

        gate = Square(
            side_length=0.7, color=YELLOW_C,
            fill_color=YELLOW_D, fill_opacity=0.25,
            stroke_width=2.5,
        ).move_to([0, -0.2, 0])
        gate_lbl = Text("U(t)", font_size=18, color=YELLOW_C)
        gate_lbl.move_to(gate)

        ctrl_dot = Dot([0, 0.2, 0], radius=0.08, color=YELLOW_C)
        ctrl_line = Line(
            [0, 0.2, 0], [0, 0.15, 0],
            stroke_width=2.5, color=YELLOW_C,
        )

        circuit = VGroup(
            anc, sys, anc_lbl, sys_lbl,
            gate, gate_lbl, ctrl_dot, ctrl_line,
        )
        circuit.move_to([x, -0.2, 0])
        self.play(Create(circuit), run_time=1.2)
        self.wait(0.3)

    def _stage3_gevp(self):
        x = self.COLS[2]
        t = self._stage_title(x, "3. GEVP", GREEN_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"H c = E S c",
            font_size=22, color=GREEN_C,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.6)

        # Convergence curve
        axes = Axes(
            x_range=[1, 8, 1], y_range=[-1.165, -1.125, 0.01],
            x_length=3.0, y_length=2.0,
            axis_config={
                "stroke_color": GREY_B,
                "font_size": 12,
                "include_tip": False,
            },
        ).move_to([x, -0.5, 0])

        qse_e = [-1.1287, -1.1553, -1.1570, -1.1600,
                 -1.1607, -1.1607, -1.1607, -1.1611]
        exact = -1.1627
        r_vals = np.arange(1, 9)

        pts = [axes.coords_to_point(r, e)
               for r, e in zip(r_vals, qse_e)]
        curve = VMobject(color=BLUE_C, stroke_width=3)
        curve.set_points_as_corners(pts)
        dots = VGroup(*[Dot(p, radius=0.045, color=BLUE_C) for p in pts])

        exact_line = DashedLine(
            axes.coords_to_point(1, exact),
            axes.coords_to_point(8, exact),
            color=RED_C, stroke_width=2,
        )
        exact_lbl = Text("exact", font_size=11, color=RED_C)
        exact_lbl.next_to(
            axes.coords_to_point(8, exact), RIGHT, buff=0.06,
        )

        self.play(Create(axes), run_time=0.4)
        self.play(Create(exact_line), FadeIn(exact_lbl), run_time=0.4)
        self.play(
            Create(curve),
            LaggedStart(*[GrowFromCenter(d) for d in dots],
                        lag_ratio=0.12),
            run_time=1.2,
        )
        self.wait(0.5)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _stage_title(self, x, text, color):
        t = Text(text, font_size=22, color=color)
        t.move_to([x, self.Y_TITLE, 0])
        return t

    def _counter(self, current: int, total: int) -> Text:
        c = Text(f"{current} / {total}", font_size=18, color=GREY_B)
        c.to_corner(UR, buff=0.3)
        return c

    def _advance_counter(self, old, current: int, total: int) -> Text:
        new = self._counter(current, total)
        self.remove(old)
        self.add(new)
        return new