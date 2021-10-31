from enum import Enum


class Policy(Enum):
    PURE = "pure"
    SEMI_PURE = "semi_pure"
    PURE_ROCK = "pure_rock"
    PURE_PAPER = "pure_paper"
    PURE_SCISSOR = "pure_scissor"

    def __str__(self):
        return self.value


POLICY_SCENARIO_MAP = {
    str(Policy.PURE): "running_with_scissors_in_the_matrix_0",
    str(Policy.SEMI_PURE): "running_with_scissors_in_the_matrix_1",
    str(Policy.PURE_ROCK): "running_with_scissors_in_the_matrix_2",
    str(Policy.PURE_PAPER): "running_with_scissors_in_the_matrix_3",
    str(Policy.PURE_SCISSOR): "running_with_scissors_in_the_matrix_4",
}
