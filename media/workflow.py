"""
Animated pipeline diagram:

    Molecule → OVOS → Qubit Hamiltonian → QSE / SQD → E_0

Renders to media/output/workflow.gif.
"""

from manim import *


class Workflow(Scene):
    """Five-stage pipeline with sequential reveal and classical/quantum grouping."""

    STAGES = [
        ("Molecule",           BLUE_C),
        ("OVOS",               GREEN_C),
        ("Qubit\nHamiltonian", YELLOW_C),
        ("QSE / SQD",          ORANGE),
        ("E\u2080",            RED_C),  # E₀
    ]

    BOX_W = 2.6
    BOX_H = 1.6
    GAP = 0.25
    Y0 = 0.5

    def construct(self):
        boxes, arrows, positions = self._build_layout()
        classical_group, quantum_group = self._build_groups(positions)

        title = Text("Quantum diagonalization pipeline", font_size=32)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.5)

        # Sequential reveal
        self.play(FadeIn(boxes[0], shift=UP * 0.3), run_time=0.5)
        self.wait(0.15)
        for i in range(1, len(boxes)):
            self.play(
                GrowArrow(arrows[i - 1]),
                FadeIn(boxes[i], shift=UP * 0.3),
                run_time=0.45,
            )
            self.wait(0.1)
        self.wait(0.4)

        # Classical/quantum grouping
        self.play(
            GrowFromCenter(classical_group[0]),
            FadeIn(classical_group[1]),
            GrowFromCenter(quantum_group[0]),
            FadeIn(quantum_group[1]),
            run_time=0.7,
        )
        self.wait(1.5)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _build_layout(self):
        n = len(self.STAGES)
        total_w = n * self.BOX_W + (n - 1) * self.GAP
        start_x = -total_w / 2 + self.BOX_W / 2

        boxes = VGroup()
        positions: list[float] = []
        for i, (label, color) in enumerate(self.STAGES):
            x = start_x + i * (self.BOX_W + self.GAP)
            positions.append(x)

            rect = RoundedRectangle(
                corner_radius=0.18,
                width=self.BOX_W, height=self.BOX_H,
                stroke_color=color, stroke_width=3.5,
                fill_color=color, fill_opacity=0.15,
            )
            rect.move_to([x, self.Y0, 0])

            text = Text(label, font_size=30, color=color)
            text.move_to(rect.get_center())
            boxes.add(VGroup(rect, text))

        arrows = VGroup()
        for i in range(n - 1):
            arrows.add(Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                buff=0.02, stroke_width=3, color=GREY_B,
                max_tip_length_to_length_ratio=0.3,
            ))
        return boxes, arrows, positions

    def _build_groups(self, positions):
        """Underline the classical and quantum halves of the pipeline."""
        y = -0.9
        half = self.BOX_W / 2

        classical_left = positions[0] - half
        classical_right = positions[2] + half
        classical_line = Line(
            [classical_left, y, 0], [classical_right, y, 0],
            color=BLUE_C, stroke_width=4,
        )
        classical_label = Text("classical", font_size=24, color=BLUE_C)
        classical_label.next_to(classical_line, DOWN, buff=0.15)

        quantum_left = positions[3] - half
        quantum_right = positions[4] + half
        quantum_line = Line(
            [quantum_left, y, 0], [quantum_right, y, 0],
            color=ORANGE, stroke_width=4,
        )
        quantum_label = Text("quantum", font_size=24, color=ORANGE)
        quantum_label.next_to(quantum_line, DOWN, buff=0.15)

        return (
            VGroup(classical_line, classical_label),
            VGroup(quantum_line, quantum_label),
        )