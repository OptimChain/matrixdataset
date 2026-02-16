<h2><a href="https://leetcode.com/problems/longest-almost-palindromic-substring/">3844. Longest Almost-Palindromic Substring</a></h2>

### Difficulty: Medium

You are given a string `s` consisting of lowercase English letters.

A substring is **almost-palindromic** if it becomes a palindrome after removing **exactly one** character from it.

Return the length of the **longest** almost-palindromic substring of `s`.

**Example 1:**

```
Input: s = "abca"
Output: 4
Explanation: The substring "abca" is almost-palindromic — removing 'c' gives "aba", which is a palindrome.
```

**Example 2:**

```
Input: s = "abba"
Output: 4
Explanation: The substring "abba" is almost-palindromic — removing one 'b' gives "aba", which is a palindrome.
```

**Example 3:**

```
Input: s = "zzabba"
Output: 5
Explanation: The substring "zabba" is almost-palindromic — removing 'z' gives "abba", which is a palindrome.
```

**Constraints:**

- `2 <= s.length <= 2500`
- `s` consists of only lowercase English letters.
