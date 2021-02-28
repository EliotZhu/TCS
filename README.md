# CDSM - Casual Inference using Deep Bayesian Dynamic Survival Models}
{Zhu and Gallego

[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![MIT
license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.1016/j.jbi.2020.103474.svg)](https://arxiv.org/abs/2101.10643)

**Authors:** [Jie Zhu](https://scholar.google.com/citations?user=Cw5v2f4AAAAJ&hl=en) and
[Blanca Gallego](https://cbdrh.med.unsw.edu.au/people/associate-professor-blanca-gallego-luxan)

Causal inference in longitudinal observational health data often requires the accurate estimation of treatment effects on time-to-event outcomes in the presence of time-varying covariates. To tackle this sequential treatment effect estimation problem, we have developed a causal dynamic survival model (CDSM) that uses the potential outcomes framework with the Bayesian recurrent sub-networks to estimate the difference in survival curves. Using simulated survival datasets, CDSM has shown good causal effect estimation performance across scenarios of sample dimension, event rate, confounding and overlapping. However, we found increasing the sample size is not effective if the original data is highly confounded or with low level of overlapping. In two large clinical cohort studies, our model identified the expected conditional average treatment effect and detected individual effect heterogeneity over time and patient subgroups. The model provides individualized absolute treatment effect estimations that could be used in recommendation systems.

Please see sample.ipynb for examples. 
