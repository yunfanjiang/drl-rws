from enum import Enum


class Policy(Enum):
    SEMI_PURE = "semi_pure"
    PURE_ROCK = "pure_rock"
    PURE_PAPER = "pure_paper"
    PURE_SCISSOR = "pure_scissor"

    def __str__(self):
        return self.value


POLICY_SCENARIO_MAP = {
    Policy.SEMI_PURE: "running_with_scissors_in_the_matrix_1",
    Policy.PURE_ROCK: "running_with_scissors_in_the_matrix_2",
    Policy.PURE_PAPER: "running_with_scissors_in_the_matrix_3",
    Policy.PURE_SCISSOR: "running_with_scissors_in_the_matrix_4",
}
