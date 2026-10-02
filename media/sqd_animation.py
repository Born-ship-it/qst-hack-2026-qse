"""
Sample-Based Quantum Diagonalization visualisation.

Four sequential stages laid out in four columns so that any paused frame
shows the full pipeline:

    1. Time-evolve  |  2. Sample  |  3. Build H  |  4. Diagonalize
"""

from manim import *
import numpy as np


class SqdAnimation(Scene):

    COLS = [-4.8, -1.6, 1.6, 4.8]
    Y_TITLE = 2.3
    Y_BODY = 0.6

    def construct(self):
        title = Text(
            "Sample-Based Quantum Diagonalization", font_size=32,
        )
        title.to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=0.6)
        self.wait(0.15)

        counter = self._counter(0, 4)
        self.add(counter)

        self._stage1_time_evolve()
        counter = self._advance_counter(counter, 1, 4)

        self._stage2_sample()
        counter = self._advance_counter(counter, 2, 4)

        self._stage3_build_h()
        counter = self._advance_counter(counter, 3, 4)

        self._stage4_diagonalize()
        counter = self._advance_counter(counter, 4, 4)

        self.wait(2.0)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage1_time_evolve(self):
        x = self.COLS[0]
        t = self._stage_title(x, "1. Time-evolve", BLUE_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle",
            font_size=22, color=BLUE_C,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.6)

        # Probability bar chart, peaked at the reference state
        probs = [0.68, 0.16, 0.08, 0.04, 0.02, 0.01, 0.01]
        bitstrings = ["0011", "0101", "1001", "0110", "1010", "1100", "1110"]
        bar_w = 0.34
        spacing = 0.42
        base_y = -2.2
        max_h = 3.0

        bars = VGroup()
        labels = VGroup()
        for i, (p, bs) in enumerate(zip(probs, bitstrings)):
            bar = Rectangle(
                width=bar_w, height=p * max_h,
                fill_color=BLUE_D, fill_opacity=0.75,
                stroke_color=BLUE_B, stroke_width=1.5,
            )
            bar.move_to([x + (i - 3) * spacing,
                         base_y + p * max_h / 2, 0])
            bars.add(bar)

            lbl = Text(bs, font_size=12, color=GREY_B)
            lbl.move_to([x + (i - 3) * spacing, base_y - 0.2, 0])
            labels.add(lbl)

        self.play(
            LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars],
                        lag_ratio=0.06),
            FadeIn(labels, shift=UP * 0.1),
            run_time=1.0,
        )
        self.wait(0.3)

    def _stage2_sample(self):
        x = self.COLS[1]
        t = self._stage_title(x, "2. Sample", GREEN_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = Text(
            "measure in Z-basis", font_size=20, color=GREEN_C,
        ).move_to([x, 1.5, 0])
        self.play(FadeIn(eq), run_time=0.4)

        # Sampled bitstrings as a tall column
        drawn = [
            "0011", "0011", "0011", "0011",
            "0101", "0101",
            "1001", "0110",
        ]
        samples = VGroup(*[
            Text(bs, font_size=20, color=GREEN_C)
            for bs in drawn
        ]).arrange(DOWN, buff=0.15)
        samples.move_to([x, -0.6, 0])

        self.play(
            LaggedStart(*[FadeIn(s, shift=LEFT * 0.4) for s in samples],
                        lag_ratio=0.1),
            run_time=1.0,
        )
        self.wait(0.3)

    def _stage3_build_h(self):
        x = self.COLS[2]
        t = self._stage_title(x, "3. Build H", YELLOW_C)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"H_{ij} = \langle b_i|H|b_j\rangle",
            font_size=22, color=YELLOW_C,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.6)

        # 4x4 grid representing the projected matrix
        n = 4
        cell = 0.55
        grid = VGroup()
        for i in range(n):
            for j in range(n):
                on_diag = (i == j)
                sq = Square(
                    side_length=cell,
                    stroke_color=YELLOW_C,
                    stroke_width=1.4,
                    fill_color=YELLOW_D,
                    fill_opacity=0.35 if on_diag else 0.12,
                )
                sq.move_to([
                    x + (j - (n - 1) / 2) * cell,
                    -0.6 + ((n - 1) / 2 - i) * cell,
                    0,
                ])
                grid.add(sq)

        self.play(Create(grid), run_time=0.9)
        self.wait(0.3)

    def _stage4_diagonalize(self):
        x = self.COLS[3]
        t = self._stage_title(x, "4. Diagonalize", ORANGE)
        self.play(FadeIn(t), run_time=0.35)

        eq = MathTex(
            r"H c = E S c",
            font_size=22, color=ORANGE,
        ).move_to([x, 1.5, 0])
        self.play(Write(eq), run_time=0.6)

        eigenvalue = MathTex(
            r"E_0 = -1.1626\,\text{Ha}",
            font_size=32, color=ORANGE,
        ).move_to([x, -0.6, 0])
        self.play(Write(eigenvalue), run_time=0.9)

        # Small pulse to highlight the answer
        self.play(
            Indicate(eigenvalue, color=ORANGE, scale_factor=1.15),
            run_time=0.7,
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