"""
Navigation estimation/integration solutions implemented in the toolbox.
This module contains all Inertial Navigation System (INS) solutions for
estimating the current states of the vehicle, specifically the estimated
position, velocity, acceleration, attitude and angle rates. All of these
solutions provide methods for taking noisy measurements from multiple
sources and using them to maintain an up-to-date estimation of these states.

As of version 1.0 of the toolbox this module contains the following solutions:
 * Standard Numerical INS (StandardINS) [Base Class]
 * Runge Kutta 4th Order Integration (RungeKutta)
 * Adams-Bashforth Integration (AdamsBashforth)
 * Kalman Filter Integration (KalmanFilter)
"""
