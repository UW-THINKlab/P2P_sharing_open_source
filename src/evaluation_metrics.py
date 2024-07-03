# Author: Zhengyang Li
# Email: zhengyang.li@connect.polyu.hk, lzy95@uw.edu
# Date: 2023-04-22
# Description: This file contains the evaluation metrics for community resilience.

import numpy as np

def get_survival_rate(resource, t):
    """
    Method:
        Given the time stamp, calculate the survival rate.
    Parameters:
        resource: np.array
            The resource array.
        t: float
            The time stamp.
    Returns:
        survival_rate: float
    """
    return np.sum(resource >= t) / len(resource)

def get_survival_curve(resource: list or np.array, time_stamps: list or np.array) -> np.array:
    """
    Method:
        Given the time stamps, calculate the survival curve.
    Parameters:
        resource: np.array
            The resource array.
        time_stamps: list or np.array
            The time stamps.
    Returns:
        survival_curve: np.array
    """
    resource = np.array(resource)  # Convert resource to a NumPy array
    survival_curve = np.sum(resource[:, np.newaxis] >= time_stamps, axis=0)
    return survival_curve

def get_expected_survival_curve(scenario_survival_curves: np.array) -> np.array:
    """
    Method:
        Given the scenario survival curves, calculate the expected survival curve.
    Parameters:
        scenario_survival_curves: np.array. row is the scenario, column is the time stamp.
            The scenario survival curves.
    Returns:
        expected_survival_curve: np.array
    """
    expected_survival_curve = np.zeros(scenario_survival_curves.shape[1])
    for i in range(0, scenario_survival_curves.shape[1]):
        expected_survival_curve[i] = np.mean(scenario_survival_curves[:, i])
    return expected_survival_curve

def get_quantile_survival_curve(scenario_survival_curves: np.array, quantile: float):
    """
    Method:
        Given the scenario survival curves, calculate the quantile survival curve.
    Parameters:
        scenario_survival_curves: np.array. row is the scenario, column is the time stamp.
            The scenario survival curves.
        quantile: float
            The quantile.
    Returns:
        quantile_survival_curve: np.array
    """
    quantile_survival_curve = np.zeros(scenario_survival_curves.shape[1])
    for i in range(0, scenario_survival_curves.shape[1]):
        quantile_survival_curve[i] = np.quantile(scenario_survival_curves[:, i], quantile)
    return quantile_survival_curve

def get_superquantile_survival_curve(scenario_survival_curves: np.array, quantile: float):
    """
    Method:
        Given the scenario survival curves, calculate the superquantile survival curve.
    Parameters:
        scenario_survival_curves: np.array. row is the scenario, column is the time stamp.
            The scenario survival curves.
        superquantile: float
            The superquantile.
    Returns:
        superquantile_survival_curve: np.array
    """
    superquantile_survival_curve = np.zeros(scenario_survival_curves.shape[1])
    for i in range(0, scenario_survival_curves.shape[1]):
        quantile_survival_curve = np.quantile(scenario_survival_curves[:, i], quantile)
        superquantile_survival_curve[i] = np.mean(scenario_survival_curves[:, i] >= quantile_survival_curve)
    return superquantile_survival_curve

def get_resilience(x:list, y:list):
    """
    Method:
        Given the lower bound, upper bound, and function values, calculate the resilience.
    Parameters:
        x: list
            The x values.
        y: list
            The function values.
    Returns:
        resilience: float
    """
    resilience = 0
    intervals = np.array(x[1:]) - np.array(x[:-1])
    func_vals = (np.array(y[:-1]) + np.array(y[1:]))/2
    resilience = np.sum(intervals * func_vals)
    return resilience

def get_resilience_loss(x:list, y:list):
    """
    Method:
        Given the lower bound, upper bound, and function values, calculate the resilience loss.
    Parameters:
        x: list
            The x values.
        y: list
            The function values.
    Returns:
        resilience_loss: float
    """
    interval = x[-1] - x[0]
    return interval * y[0] - get_resilience(x, y)