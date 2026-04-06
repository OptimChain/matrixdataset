from typing import List


class Solution:
    def minIncrease(self, nums: List[int]) -> int:
        n = len(nums)
        if n % 2:
            # Odd length: only one valid alignment — peaks at odd indices 1, 3, 5, ..., n-2
            dp = 0
            for i in range(1, n - 1, 2):
                dp += max(max(nums[i - 1], nums[i + 1]) - nums[i] + 1, 0)
            return dp

        # Even length: two possible alignments per pair of consecutive positions.
        # dp1 = cost of pure odd-index peaks (1, 3, 5, ...)
        # dp2 = optimal cost allowing switches from odd to even alignment.
        #        Once even, must stay even (adjacent constraint).
        #        dp1 can transition to dp2 at any step, but not vice versa.
        dp1, dp2 = 0, 0
        for i in range(1, n - 2, 2):
            dp1 += max(max(nums[i - 1], nums[i + 1]) - nums[i] + 1, 0)
            dp2 += max(max(nums[i], nums[i + 2]) - nums[i + 1] + 1, 0)
            dp2 = min(dp1, dp2)
        return dp2
