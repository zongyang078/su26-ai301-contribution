# Contribution 1: More EpochMetric's compute_fn output types

**Contribution Number:** 1  
**Student:** Zongyang Li  
**Issue:** https://github.com/pytorch/ignite/issues/1757  
**Status:** [Phase IV] [Complete]

---

## Why I Chose This Issue

I chose this issue because it directly improves PyTorch Ignite's EpochMetric class, which is a core component used in ML training pipelines. The task is well-scoped: extending compute_fn to support more output types beyond scalars. The maintainer has already provided implementation hints and a reference to Detectron2's code, making it 
a good starting point for contributing to a real ML framework. As someone targeting AI Engineer roles, working on PyTorch's ecosystem is directly relevant to my career goals.

---

## Understanding the Issue

### Problem Description

EpochMetric._result is typed as float, and compute() only handles scalar outputs. When compute_fn returns a tensor, tuple, or mapping, the code either fails silently or causes a type error downstream.

### Expected Behavior

EpochMetric should support compute_fn returning scalars, tensors, tuple/list of tensors, and mapping of tensors, with a clear TypeError for unsupported types.

### Current Behavior

compute_fn can only return a scalar (int or float). Returning other types causes silent failures or confusing downstream errors.

### Affected Components

- ignite/metrics/epoch_metric.py (EpochMetric class, compute method)
- tests/ignite/metrics/test_epoch_metric.py

---

## Reproduction Process

### Environment Setup

- OS: macOS
- Python: 3.10.20 (conda env: ignite-dev)
- PyTorch: 2.12.0
- pytorch-ignite: 0.6.0 (installed with pip install -e)
- pytest: 9.0.3

### Steps to Reproduce

1. Create EpochMetric with a compute_fn that returns a tensor
2. Call compute()
3. Result: no TypeError raised but downstream code fails silently
   because _result is typed as float only

### Reproduction Evidence

- **Commit showing reproduction:** N/A — reproduced by code inspection. epoch_metric.py line shows _result: float | None and compute() -> float, confirming the limitation described in the issue.
- **Screenshots/logs:** [If applicable]
- **My findings:** The _result type annotation and compute() return type were both float, and there was no type validation on compute_fn output. apply_to_type already existed in ignite/utils.py and could be reused for distributed broadcasting.

---

## Solution Approach

### Analysis

The root cause is in compute() method of EpochMetric:
1. _result is typed as float | None, limiting output to scalars
2. No type validation on compute_fn output
3. Distributed broadcasting only handled float scalars

The fix requires extending type annotations, adding validation, and using the existing apply_to_type utility for non-scalar broadcasting.

### Proposed Solution

- Extended _result type to support float, torch.Tensor, Sequence, Mapping
- Added type validation in compute() with clear TypeError message
- Used existing apply_to_type utility for distributed broadcasting
- Updated docstring to document supported output types
- Added 3 new tests covering tensor, tuple, and invalid output types

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** EpochMetric.compute_fn only supports scalar outputs, but users need tensor/tuple/mapping support.

**Match:** ignite/utils.py already has apply_to_type that recursively applies a function to tensors inside any nested structure.

**Plan:** 
1. Extend _result type annotation in reset()
2. Update compute() return type and add type validation
3. Use apply_to_type for distributed broadcasting of non-scalar results
4. Update docstring
5. Add tests for new output types

**Implement:** https://github.com/zongyang078/ignite/tree/feature/epoch-metric-output-types

**Review:** Follows existing code style, uses existing utilities, adds tests for all new cases.

**Evaluate:** Run pytest tests/ignite/metrics/test_epoch_metric.py - all 13 tests pass.

---

## Testing Strategy

### Unit Tests

- [x] Test case 1: compute_fn returning a tensor (per-class MSE)
- [x] Test case 2: compute_fn returning a tuple of tensors (MSE + MAE)
- [x] Test case 3: compute_fn returning invalid type (str) raises TypeError

### Integration Tests

- [ ] N/A for this change — existing test_distrib_integration covers the distributed scenario and passes with our changes

### Manual Testing

Ran pytest tests/ignite/metrics/test_epoch_metric.py: 13 passed, 9 skipped (no CUDA/GPU), 1 pre-existing error (gloo_cpu requires pytest-xdist, unrelated to our changes).

---

## Implementation Notes

### Week 1 Progress (Phase I)

- Explored issue candidates from CodePath's list
- Selected ignite issue #1757 based on maintainer guidance and 
  relevance to AI Engineer career goals
- Forked pytorch/ignite repository
- Set up local dev environment (conda ignite-dev, Python 3.10)

### Week 2 Progress (Phase II&III)

- Identified affected files using grep
- Read maintainer's implementation hints in PR #1700
- Extended EpochMetric to support tensor/tuple/mapping outputs
- Discovered str is a subclass of Sequence, added explicit str exclusion
- Added 3 new tests, all passing
- Updated docstring

### Week 3 Progress (Phase IV)
- Submitted PR #3789 to pytorch/ignite
- PR is open and awaiting maintainer review
- CI checks pending maintainer approval

### Code Changes

- **Files modified:** ignite/metrics/epoch_metric.py, tests/ignite/metrics/test_epoch_metric.py
- **Key commits:** 
  - [feat: extend EpochMetric](https://github.com/zongyang078/ignite/commit/2f12622f)
  - [test: add tests](https://github.com/zongyang078/ignite/commit/c6e82c06)
  - [docs: update docstring](https://github.com/zongyang078/ignite/commit/7c185831)
- **Approach decisions:** Used existing apply_to_type utility instead of reimplementing recursive tensor handling

---

## Pull Request

**PR Link:** https://github.com/pytorch/ignite/pull/3789

**PR Description:** Extended EpochMetric to support tensor, tuple/list, and mapping outputs from compute_fn. Added type validation with clear TypeError for unsupported types. Reused apply_to_type for distributed broadcasting. Added 3 new tests, updated docstring.

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review]

---

## Learnings & Reflections

### Technical Skills Gained

- Learned how Python's type system works (str is a subclass of Sequence)
- Learned how to use git stash to isolate changes for testing
- Understood how distributed broadcasting works in ignite with apply_to_type
- Practiced reading and contributing to a large real-world ML codebase

### Challenges Overcome

- Discovered that str passes isinstance check for Sequence, required 
  explicit exclusion — caught by a failing test
- gloo_cpu test error turned out to be a pre-existing environment issue, 
  confirmed by testing on original codebase with git stash

### What I'd Do Differently Next Time

- Set up the dev environment earlier before selecting the issue
- Read CONTRIBUTING.md more carefully before writing the first commit

---

## Resources Used

- [ignite CONTRIBUTING.md](https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md)
- [Maintainer's implementation hints in PR #1700](https://github.com/pytorch/ignite/pull/1700#issuecomment-790413236)
- [Detectron2 comm.py reference](https://github.com/facebookresearch/detectron2/blob/7f8f29deae278b75625872c8a0b00b74129446ac/detectron2/utils/comm.py#L148)
- [ignite issue #1757](https://github.com/pytorch/ignite/issues/1757)
