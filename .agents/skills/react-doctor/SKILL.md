---
name: react-doctor
description: "Scans React codebase for security, performance, correctness, and architecture issues. Outputs a 0-100 score with actionable diagnostics. Use when: (1) After making React changes to catch issues early, (2) Reviewing code before commit, (3) Finishing a feature or fixing bugs in a React project. Triggers: react-doctor, react doctor, scan react, react health, react quality."
metadata:
  author: millionco
  version: "1.0"
---

# React Doctor

Scans React codebase for issues. Outputs a 0-100 score with actionable diagnostics.

## Quick Navigation

| Situation | Go To |
|-----------|-------|
| Just finished React changes | [When to Run](#when-to-run) |
| Got a score, what now? | [Score Interpretation](#score-interpretation) |
| Need to fix specific warnings | [Common Issues & Fixes](#common-issues--fixes) |
| Setting up CI gates | [CI Integration](#ci-integration) |
| Agent workflow integration | [Agent Workflow](#agent-workflow) |

---

## When to Run

**ALWAYS run after:**
- Making changes to React components
- Adding new hooks or useEffect patterns
- Modifying page.tsx or layout.tsx files
- Before committing React-related changes

**Commands:**

```bash
npm run doctor          # Scan changed files (diff mode in worktrees)
npm run doctor:score    # Output only the score (for CI)
npm run doctor:diff     # Diff mode against main branch
npm run doctor:full     # Full scan with lint + dead code
```

**Diff Mode:** Automatically enabled when in a git worktree. Scans only changed files vs main branch.

---

## Score Interpretation

| Score | Status | Agent Action |
|-------|--------|--------------|
| **90-100** | Pass | Proceed. No action required. |
| **0-89** | Fail | MUST fix all issues before committing. |
**Decision Matrix:**

| Score | Can Commit? | Action |
|-------|-------------|--------|
| ≥90 | ✅ Yes | None required |
| <90 | ❌ No | Fix ALL issues, re-run until ≥90 |
---

## Common Issues & Fixes

### State & Effects

| Warning | Fix |
|---------|-----|
| Multiple setState in useEffect | Use `useReducer` or derive state from props |
| Missing dependency in useEffect | Add to dependency array, or use `useCallback`/`useMemo` |
| useSearchParams without Suspense | Wrap: `<Suspense fallback={<Skeleton />}><Component /></Suspense>` |

### Performance

| Warning | Fix |
|---------|-----|
| Index used as key | Use stable ID: `key={item.id}` — index keys break on reorder/filter |
| Inline render function | Extract to named component for proper reconciliation |
| Scroll listener without passive | `addEventListener('scroll', handler, { passive: true })` |

### Architecture

| Warning | Fix |
|---------|-----|
| Component too large (>200 lines) | Extract logical sections into focused components |
| Too many useState calls (>5) | Group related state with `useReducer` |

### Next.js

| Warning | Fix |
|---------|-----|
| Page without metadata | Add `export const metadata = { title: '...', description: '...' }` |
| Using `<img>` instead of next/image | `import Image from 'next/image'` — auto WebP/AVIF, lazy loading |
| useSearchParams without Suspense | Wrap with `<Suspense>` boundary |

### Correctness

| Warning | Fix |
|---------|-----|
| Inline render function | Extract to named component |
| Conditional hooks | Move hooks to top level, use early returns for rendering |

---

## Agent Workflow

### After Making React Changes

```
1. Run: npm run doctor
2. Read score and warnings
3. If score < 90: Fix ALL issues, re-run
4. If score ≥ 90: Proceed to commit
```

### Before Committing React Code

```
1. Run: npm run doctor
2. If warnings exist:
   - Quick fix (<2 min): Fix now
   - Complex fix: Note in commit, create follow-up task
3. Include score in commit message if notable
```

### CI Integration

For CI pipelines, use the score-only output:

```bash
npm run doctor:score
# Returns single number: 93
```

**CI Gate Example:**
```bash
SCORE=$(npm run -s doctor:score)
if [ "$SCORE" -lt 90 ]; then
  echo "React Doctor score too low: $SCORE (minimum: 90)"
  exit 1
fi
```

---

## Anti-Patterns (Agent-Specific)

| Pattern | Problem | Fix |
|---------|---------|-----|
| **Score Chasing** | Optimizing for score over actual quality | Fix real issues, ignore cosmetic warnings |
| **Warning Blindness** | Ignoring warnings because score is high | Review ALL warnings, fix high-impact ones |
| **One-Shot Scanning** | Running once and never again | Run after every React change |

---

## Invariants (Patterns for 90+ Scores)

**Follow these patterns to prevent regressions and maintain 90+ scores:**

| Pattern | Invariant | Why |
|---------|-----------|-----|
| **List keys** | ALWAYS use stable keys (`item.id`), NEVER index | Index keys break on reorder/filter |
| **useSearchParams** | ALWAYS wrap with `<Suspense>` | Without it, entire page bails to CSR |
| **Scroll/resize listeners** | ALWAYS add `{ passive: true }` | Blocking listeners hurt scroll perf |
| **Component size** | Keep under 200 lines; extract sections | Large components are hard to maintain |
| **Related state** | Group >5 useState into useReducer | Prevents race conditions, easier to debug |
| **Next.js pages** | Add `metadata` export; use `next/image` | SEO + automatic optimization |

### Quick Fixes (Highest Impact / Lowest Effort)

These issues appear most often and are quick to fix:

| Issue | Fix | Time |
|-------|-----|------|
| Index key | `key={item.id}` | ~1 min |
| useSearchParams w/o Suspense | Wrap with `<Suspense>` | ~2 min |
| Scroll listener | `{ passive: true }` | ~1 min |
| Missing metadata | Add `export const metadata` | ~1 min |
| `<img>` tag | Use `next/image` | ~2 min |

---

## Config File

Project uses `react-doctor.config.json`:

```json
{
  "verbose": true,
  "ignore": {
    "rules": [],
    "files": ["src/components/ui/**"]
  }
}
```

**Why ignore UI components?** shadcn/ui components are third-party. Scanning them wastes time and produces noise.

---

## Integration with Other Skills

| Task | Skill | Usage |
|------|-------|-------|
| After fixing issues | `code-quality` | Run typecheck + lint |
| Committing | `git-commit` | Include score if notable |
| TDD workflow | `tdd` | Run doctor after tests pass |

---

## Rule Categories Reference

| Category | What It Checks |
|----------|----------------|
| State & Effects | useState/useEffect patterns, dependency arrays |
| Performance | Re-renders, memoization, event listeners |
| Architecture | Component size, prop drilling |
| Security | dangerouslySetInnerHTML, XSS |
| Correctness | Key usage, conditional hooks, race conditions |
| Accessibility | ARIA, focus management, semantic HTML |
| Next.js | Metadata, Suspense boundaries, next/image |
| Dead Code | Unused exports, duplicate code |

---

## References

- [react-doctor](https://github.com/millionco/react-doctor) - Official repo
- [React Doctor](https://www.react.doctor) - Documentation
