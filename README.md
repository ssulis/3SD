`` Semi-supervised standardized detection (3SD) of extrasolar planets ''
 
 The directory contains an implementation of Algorithms 1 (3SD) and 2 (Bootstrap), as described in Sulis et al., 2021 (submitted to A&A).
 
 It also contains two examples of how to run the detection algorithms:
 
 - ``Example1_NTS_available.py'', which implements the case where a null training sample (NTS) of the stochastic noise is available. 
 
 and
 
 - ``Example2_no_NTS_available.py'', which implements the case where no NTS of the stochastic noise is available. In this case, the NTS is estimated from the RV series under test. 
 
The different steps are detailed in the python codes.

For practical implementation, we note that the procedure is versatile in the sense that the specific couple (test, periodogram) is left to the user, and the procedure may adapt to different noise sources, null training samples (if available), and time sampling grids. 

 
