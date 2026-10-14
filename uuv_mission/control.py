"""Discrete-time PD control for the UUV's depth."""


class PDController:
    """Keep the previous error between calls; reset before a new simulation."""

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.reset()

    def reset(self) -> None:
        """Restore the initial condition e[-1] = 0."""
        self.previous_error = 0.0

    def __call__(self, reference: float, observation: float) -> float:
        """Return the action for the target and observed depths."""
        error = reference - observation
        action = self.kp * error + self.kd * (error - self.previous_error)
        self.previous_error = error
        return action
