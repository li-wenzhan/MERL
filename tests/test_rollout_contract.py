import unittest
import torch

from merl.rollout_contract import valid_response_tokens


class RolloutContractTests(unittest.TestCase):
    def test_units_are_environment_steps_not_policy_calls(self):
        responses = torch.ones(3, 4, 56, dtype=torch.long)
        actual = valid_response_tokens(responses, torch.tensor([32, 11, 0]), 8)
        torch.testing.assert_close(actual, torch.tensor([224, 77, 0]))

    def test_dummy_suffix_and_interior_hole(self):
        responses = torch.ones(2, 4, 56)
        actual = valid_response_tokens(responses, torch.tensor([32, 32]), 8,
                                       torch.tensor([[False, False, True, True], [False, True, False, False]]))
        torch.testing.assert_close(actual, torch.tensor([112, 56]))

    def test_finish_cap_and_invalid_token_ratio(self):
        torch.testing.assert_close(valid_response_tokens(torch.ones(1, 2, 56), torch.tensor([99]), 8),
                                   torch.tensor([112]))
        with self.assertRaises(ValueError):
            valid_response_tokens(torch.ones(1, 2, 55), torch.tensor([16]), 8)


if __name__ == "__main__":
    unittest.main()
