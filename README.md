SINGLE --- A code for fitting single-transit events using stellar density information
---

The usage of the code is very simple. Do:

   ``python single.py -lcfile PATHTOFILE -sdmean MEAN_STELLAR_DENSITY -sdsigma SDERROR -t0mean MEAN_TIME_OF_TRANSIT -t0sigma T0ERROR -ldlaw LDLAW``

Where: 
 
   ``PATHTOFILE``               Path to the lightcurve, expected to have in the first column time and in the second relative fluxes.

   ``MEAN_STELLAR_DENSITY``     Value of the mean stellar density, in kg/cm^3.

   ``SDERROR``                  Error on the mean stellar density.

   ``MEAN_TIME_OF_TRANSIT``     Mean time-of-transit center

   ``T0ERROR``                  Error on the mean time-of-transit center

   ``LDLAW``                    The limb-darkening law to use (can be ``linear`` or any two-parameter law from: ``quadratic``, 
                                ``square-root`` and ``logarithmic``).

Optionally, you can set ``--resampling`` if you want to apply Kepler-like resampling. You can modify the integration time of the resampling by 
using ``-texp NUMBER`` where ``NUMBER`` is the exposure time in days (default is ``0.020434`` days, which is the Kepler long-cadence effective 
exposure time), and the ammount of resampling points by specifying it with the variable ``-nresampling N``, with ``N`` the number of points to 
resample (default is 20).
