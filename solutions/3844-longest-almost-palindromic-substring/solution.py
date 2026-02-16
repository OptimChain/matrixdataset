class Solution:
    def longestAlmostPalindromicSubstring(self, s: str) -> int:
        n = len(s)

        # Precompute palindrome table: is_pal[i][j] = True if s[i..j] is a palindrome
        is_pal = [[False] * n for _ in range(n)]
        for i in range(n):
            is_pal[i][i] = True
        for i in range(n - 1):
            is_pal[i][i + 1] = (s[i] == s[i + 1])
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

        ans = 0

        # For each substring s[i..j], check if removing one character makes it a palindrome
        for i in range(n):
            for j in range(i + 1, n):  # length >= 2 (after removal, at least 1 char)
                length = j - i + 1
                if length <= ans:
                    continue
                # Check removing each endpoint or using two-pointer approach
                # Removing s[i]: check if s[i+1..j] is palindrome
                if is_pal[i + 1][j]:
                    ans = length
                    continue
                # Removing s[j]: check if s[i..j-1] is palindrome
                if is_pal[i][j - 1]:
                    ans = length
                    continue
                # Removing an interior character: use two-pointer
                left, right = i, j
                found = False
                while left < right:
                    if s[left] == s[right]:
                        left += 1
                        right -= 1
                    else:
                        # Try skipping left or right
                        if is_pal[left + 1][right] or is_pal[left][right - 1]:
                            found = True
                        break
                else:
                    # The substring itself is a palindrome (odd-length center),
                    # removing the center character still yields a palindrome
                    found = True
                if found:
                    ans = length

        return ans
