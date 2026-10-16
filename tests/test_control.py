import unittest

from uuv_mission.control import PDController


class PDControllerTests(unittest.TestCase):
    def test_first_action_uses_zero_previous_error(self):
        controller = PDController()
        self.assertAlmostEqual(controller(reference=5.0, observation=3.0), 1.5)

    def test_unchanged_error_has_zero_derivative_term(self):
        controller = PDController()
        controller(reference=5.0, observation=3.0)
        self.assertAlmostEqual(controller(reference=5.0, observation=3.0), 0.3)

    def test_actions_follow_changing_and_negative_errors(self):
        controller = PDController()
        # Errors are 2, 1, -1, and 0, in that order.
        for observation, expected in [(3.0, 1.5), (4.0, -0.45),
                                      (6.0, -1.35), (5.0, 0.6)]:
            with self.subTest(observation=observation):
                self.assertAlmostEqual(controller(5.0, observation), expected)

    def test_reset_restores_first_action(self):
        controller = PDController()
        controller(5.0, 3.0)
        controller(5.0, 4.0)
        controller.reset()
        self.assertAlmostEqual(controller(5.0, 3.0), 1.5)

    def test_controller_instances_have_independent_history(self):
        first = PDController()
        second = PDController()
        first(5.0, 3.0)
        self.assertAlmostEqual(second(5.0, 3.0), 1.5)
        self.assertAlmostEqual(first(5.0, 3.0), 0.3)


if __name__ == "__main__":
    unittest.main()
