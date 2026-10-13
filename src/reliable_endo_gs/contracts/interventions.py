"""Versioned offline/predicted intervention outcome identities, not deployment GT.

Ordered action names and component units are shared by collection, fitting and
selection. The Gaussian action's stereo-depth changes are structurally zero.
"""

COMPONENT_NAMES = (
    "global_depth_mae_gain_mm",
    "local_depth_mae_gain_mm",
    "global_depth_rmse_gain_mm",
    "local_depth_rmse_gain_mm",
    "global_depth_p95_gain_mm",
    "left_psnr_gain_db",
    "right_psnr_gain_db",
    "left_ssim_gain",
    "right_ssim_gain",
    "outside_rgb_gain",
    "local_rgb_gain",
    "geometry_coverage_change",
)

# Disjoint, exhaustive event taxonomy. Outside RGB is collateral/global.
LOCAL_RISK_COMPONENT_NAMES = (
    "local_depth_mae_gain_mm",
    "local_depth_rmse_gain_mm",
    "local_rgb_gain",
)
RISK_EVENT_NAMES = ("local", "global", "any")


ACTION_NAMES = ("correspondence", "learned_stereo", "gaussian")
