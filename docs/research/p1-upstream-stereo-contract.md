# Phase-I P1 upstream stereo contract

This note records the read-only contract audited from pinned Endo-E2E-GS
commit `186fa2b4a2159b28393492f6df1aa444b54391a8`.

## Forward path

`StereoEndoModel.forward(..., is_train=False)` concatenates the normalized
left and right images along the batch dimension, encodes them with
`img_encoder`, and passes the resulting third feature level to
`raft_stereo`. The pinned Stage-2 configuration uses three validation
iterations. With `test_mode=True`, upstream returns only the final disparity;
with `test_mode=False`, `RAFTStereo` returns all three recurrent predictions.

Each retained prediction is full-resolution `[B, 1, H, W]` after the pinned
convex upsampling path. The final prediction is left-reference disparity in
pixels, with the verified convention `x_left - x_right > 0`.

The baseline runner disables mixed precision and uses FP32. P1 uses the same
resolved setting and does not invoke the Gaussian renderer.

## Available evidence

| Evidence | Contract | P1 use |
| --- | --- | --- |
| Recurrent disparities | three aligned `[B,1,H,W]` maps when `test_mode=False` | U1, U2 |
| Right-reference disparity | not returned by the normal forward | U3 requires a swapped-image second forward |
| Correlation tensor | internal feature correlation only; no normalized probability distribution | U5 unavailable |
| Recurrent hidden/update tensors | not exposed at the model boundary | not used |
| Rectified images | normalized `[-1,1]` left/right tensors | U4 |

## Proxy conventions

- U1 is `abs(d_K - d_(K-1))`, in disparity pixels.
- U2 is population standard deviation over the retained iteration stack,
  `unbiased=False`, in disparity pixels.
- U3, when explicitly enabled, runs the same model on `(right, left)`. Its
  output is right-reference disparity, so consistency is
  `abs(d_L(x) + d_R(x - d_L(x)))` with out-of-view masking.
- U4 samples the rectified right image at `x - d_L` and reports mean-channel
  absolute residual in normalized image units. It is not optimized.
- U5 is unavailable because the pinned correlation implementation exposes
  feature correlations, not a probability-like distribution with a justified
  entropy axis.

P1 validity maps are derived from finite, strictly positive predicted
disparity and sampling support. Ground-truth disparity and its mask are used
only to form oracle error targets after proxy extraction.
