`` Semi-supervised standardized detection (3SD) of extrasolar planets ''
 
 The directory contains an implementation of the Algorithms 1 (detection) and 2 (p-value computation), as described in Sulis et al. 2022 (A&A 667, A04, 2022, https://arxiv.org/abs/2207.03740).
 
 It also contains two examples of how to run the detection algorithms:
 
 - ``Example1_NTS_available.py'', which implements the case where a null training sample (NTS) of the stochastic noise is available (see Sec. 5.3). 
 
 and
 
 - ``Example2_no_NTS_available.py'', which implements the case where no NTS of the stochastic noise is available. In this case, the NTS is estimated from the RV series under test (see Sec. 5.5). 
 
The different steps are detailed in the python codes.

For practical implementation, we note that the procedure is versatile in the sense that the specific couple (test, periodogram) is left to the user, and the procedure may adapt to different noise sources, null training samples (if available), and time sampling grids. 

To run the codes, you need Python3 and the following libraries:
>> numpy, matplotlib, astropy, lmfit, PyAstronomy, math, tqdm, statsmodels

If you have any suggestion, please write me a message at sophia.sulis@lam.fr.
