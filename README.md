# Causal inference for observational longitudinal studies using deep survival models


[![Project Status: Active – The project has reached a stable, usable
state and is being actively
developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![MIT
license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.1016/j.jbi.2020.103474.svg)](https://arxiv.org/abs/2101.10643)

**Authors:** [Jie Zhu](https://scholar.google.com/citations?user=Cw5v2f4AAAAJ&hl=en) and
[Blanca Gallego](https://cbdrh.med.unsw.edu.au/people/associate-professor-blanca-gallego-luxan)


Please see Causal Inference.ipynb for examples of the proposed model. 



Causal inference for observational longitudinal studies often requires the accurate estimation of treatment effects on time-to-event outcomes in the presence of time-dependent patient history and time-dependent covariates. 

To tackle this longitudinal treatment effect estimation problem, we have developed a time-variant causal survival (TCS) model that uses the potential outcomes framework with an ensemble of recurrent subnetworks to estimate the difference in survival probabilities and its confidence interval over time as a function of time-dependent covariates and treatments. 

Using simulated survival datasets, the TCS model showed good causal effect estimation performance across scenarios of varying sample dimensions, event rates, confounding and overlapping. However, increasing the sample size was not effective in alleviating the adverse impact of a high level of confounding. In a large clinical cohort study, TCS identified the expected conditional average treatment effect and detected individual treatment effect heterogeneity over time. TCS provides an efficient way to estimate and update individualized treatment effects over time, in order to improve clinical decisions.

The use of a propensity score layer and potential outcome subnetworks helps correcting for selection bias. However, the proposed model is limited in its ability to correct the bias from unmeasured confounding, and more extensive testing of TCS under extreme scenarios such as low overlapping and the presence of unmeasured confounders is desired and left for future work. 

TCS fills the gap in causal inference using deep learning techniques in survival analysis. It considers time-varying confounders and treatment options. Its treatment effect estimation can be easily compared with the conventional literature, which uses relative measures of treatment effect. We expect TCS will be particularly useful for identifying and quantifying treatment effect heterogeneity over time under the ever complex observational health care environment.
