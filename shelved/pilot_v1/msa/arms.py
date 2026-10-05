"""Arm definitions. Nine arms: tied dot product, seven L^p arms, untied dot product.
NULL_ARM ('uniform') is an extra control outside the nine: a constant score, so every allowed key gets the same weight."""
import math

ARMS = ["dot_tied", "lp_1", "lp_1.5", "lp_2", "lp_3", "lp_4", "lp_8", "lp_inf", "dot_untied"]
LP_ARMS = [a for a in ARMS if a.startswith("lp_")]
NULL_ARM = "uniform"     # control outside the nine arms: constant score, so attention is uniform over the allowed keys


def parse_arm(arm):
    """Returns (kind, p, tied)."""
    if arm == "dot_tied":
        return "dot", None, True
    if arm == "dot_untied":
        return "dot", None, False
    if arm == NULL_ARM:
        return "null", None, True
    if arm.startswith("lp_"):
        s = arm[3:]
        p = math.inf if s == "inf" else float(s)
        return "lp", p, True
    raise ValueError(f"unknown arm {arm}")


def arm_label(arm):
    return {"uniform": "uniform (null)", "dot_tied": "dot (tied)", "dot_untied": "dot (untied)", "lp_inf": "L-inf"}.get(arm, "L^" + arm[3:] if arm.startswith("lp_") else arm)
