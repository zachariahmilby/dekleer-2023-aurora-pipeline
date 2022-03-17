# KHAN: Keck HIRES Aurora pipeliNe

> "You are in a position to demand nothing. I, on the other hand, am in a 
> position to grant nothing."<br>
> Khan Noonien Singh<br>
> <em>Star Trek II: The Wrath of Khan (1982)</em>

## Introduction
I'm writing this README text in March 2022. For some reason, despite being one 
of the most productive instruments at the Keck observatory on Mauna Kea, the 
HIRES spectrograph does not have a standard data reduction pipeline. There are 
some exoplanet-specific pipelines which aren't really public, a limited-use 
automatic one that doesn't produce science-quality results (and has failed on 
all of my recent data sets) and some very old IDL-based pipelines that also no 
longer seem to work and are no longer supported anymore by their authors. 
Similarly, the ``PypeIt`` package[^1], intended by its designers to be the 
standard spectroscopic data reduction pipeline for Python, does not yet have 
HIRES support. Like the package's namesake I feel vengeful.

This repository holds my attempt at making an automated pipeline for reducing 
Keck HIRES[^2] echelle spectrograph observations of aurora on the Galilean 
satellites, a project I work on at Caltech with Katherine de Kleer, Mike Brown
and Maria Camarca (also at Caltech) and Carl Schmidt (at Boston University). 
I've cobbled together many ideas and components from a variety of sources 
including the Mauna Kea Echelle Extraction ``MAKEE`` package[^3] and the 
``HIRedux`` IDL reduction pipeline [^4] written by Jason X. Prochaska.

The pipeline will:
1. Detect each order and its bounds, including those crossing between
   detectors,
2. Extract each order from the bias, flat, arc and science frames,
3. Reduce the science frames by subtracting the bias, flat-fielding and
   gain-correcting,
4. Automatically calculate a wavelength solution for each extracted order, and
5. Save the result for easy access and/or archiving (probably as a FITS file 
   since that's the same format as the original data).

I don't really know if this will work without modification for other types of 
HIRES data (I imagine the use of different filters will change how it 
operates). I've made this repository public so anyone can take and modify what 
I've done here.

[^1]: https://pypeit.readthedocs.io/en/release/
[^2]: https://www2.keck.hawaii.edu/inst/hires/
[^3]: https://sites.astro.caltech.edu/~tb/makee/
[^4]: https://www.ucolick.org/~xavier/HIRedux/

## Installation
I've used this code successfully on a Mac running macOS Monterey 
(version 12.2.1) using an Anaconda virtual environment running Python 3.10.0. 
I've also written a few unit tests which Github will automatically run to see 
if the current version runs on their "latest" version of macOS (though this 
probably isn't the latest commercial release). In any case, if I were you, I 
would install this in a virtual environment running Python 3.10 (or newer) so 
you don't mess up any of your other projects.

Here are some installation instructions for the average Anaconda user, if 
you're more advanced I'm sure you can figure it out from here.
1. \[Optional/Recommended\] Create a virtual environment (I've named it
   `myenv` in the example):<br>
   `% conda create --name myenv python=3.10`
2. Activate your virtual environment:<br>
    `% conda activate myenv`
3. Install the `khan` package and its dependencies:<br>
    `% python -m pip install git+https://github.com/zachariahmilby/keck-hires-aurora-pipeline.git`

You're now ready to use the `khan` package! Khaaan!

## Getting Started
There are three steps to running this pipeline.
1. Sort your data files manually so it knows where to find file type. 
   Unfortunately FITS headers are extremely unreliable for recording what the 
   telescope was actually looking at, so you really can't trust them. 
2. Run the data-reduction pipeline.
3. Extract the aurora brightnesses from the reduced data.

See below for details on each of these steps.

### Sorting Your Data
The only particular requirement in place for running the pipeline is to 
pre-sort your data according to the hierarchy structure below. There can be 
multiple sub-directories containing science data, and they don't have to be 
named ``science``, but the others should be named `bias`, `flat`, `arc`, 
`trace` and `flux_calibration`. For instance, you might have taken data from 
two different moons on the same night with the same detector setup (so the 
calibration files work for both), so you put your Ganymede science data in a 
directory called ``ganymede`` and your Callisto science data in a directory 
called ``callisto``. You can then point to these specifically as science data 
directories. The data file names themselves can be in any format (they'll 
probably be in either the non-unique ``hires####.fits``style or the unique 
``HI.YYYYMMDD.#####.fits.gz`` style).

```
data_directory
|
└───bias
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│   
└───flat
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│   
└───arc
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│
└───trace
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│
└───flux_calibration
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
│
└───science
│   │   file01.fits.gz
│   │   file02.fits.gz
│   │   ...
```

### Running the Pipeline
The function `run_pipeline` takes just a few inputs in order to automatically
run. I have tested it on my Ganymede data, and it works for single-detector
pre-mosaic legacy data and mosaic data in both 2x1 and 3x1 binning.

```
>>> from khan import reduce_data
>>> reduce_data(science_target_name='Ganymede', guide_satellite_name='Europa', 
                source_data_path=selected_data_path, save_path=reduced_data_path, 
                quality_assurance=False)
```

The target and guide satellite names are important for properly querying 
JPL's Horizons Ephemeris Service. The `source_data_path` points to the 
directory into which you pre-sorted your data files. The `save_path` points to
where you want the reduced data files (and quality-assurance graphics) saved.

Speaking of quality-assurance graphics, the `quality_assurance` keyword lets
you choose whether or not the pipeline saves graphics along the way so you can
check to see if anything failed or otherwise worked poorly. These graphics 
include:
1. Images of the master bias, master flat, master arc and master trace, 
2. The detected edges of the orders
3. Sets of images showing each of the science observations (Jupiter 
   calibration, guide satellite and target satellite) at each stage of
   instrument correction:
   1. Raw
   2. After cosmic-ray correction
   3. After bias-subtraction
   4. After flat-field correction, and
   5. After gain correction, and
4. The third-degree polynomial wavelength solutions for each order and the
   residuals of the fit.

> **WARNING**<br>
> Saving quality-assurance graphics substantially increases the 
> runtime of the pipeline, and due to a memory leak in Matplotlib, it may put 
> too much memory pressure on your system and cause a SystemExit. I found I 
> could run individual observation nights one at a time, but I could not reduce
> multiple nights using a loop.

Once the pipeline has finished reducing the data, you should have two FITS 
files in the `save_path` directory you specified: `flux_calibration.fits.gz`
and `science_observations.fits.gz`.

#### Flux Calibration Data

The `flux_calibration.fits.gz` file has the following structure:
```
>>> from astropy.io import fits
>>> with fits.open('flux_calibration.fits.gz') as hdul:
>>>     hdul.info()

Filename: flux_calibration.fits.gz
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU      28   (4096, 20, 32, 5)   float64   
  1  BIN_CENTER_WAVELENGTHS    1 ImageHDU         9   (4096, 32)   float64   
  2  BIN_EDGE_WAVELENGTHS    1 ImageHDU         9   (4097, 32)   float64   
  3  ECHELLE_ORDERS    1 BinTableHDU     11   32R x 1C   [I]   
  4  OBSERVATION_INFORMATION    1 BinTableHDU     18   5R x 4C   [25A, D, 19A, E]   
```
The `PRIMARY` extension holds the reduced Jupiter flux calibration data.
`BIN_CENTER_WAVELENGTHS` are the wavelength solution values for the center of
each spectral bin. Similarly, `BIN_EDGE_WAVELENGTHS` are the solution values
for the edges of each bin (in case you wanted to use 
`matplotlib.pyplot.pcolormesh` to display an image with a horizontal wavelength
axis). `ECHELLE_ORDERS` holds the actual echelle order number for each of the
orders included in the data and the wavelength solutions. Finally, the 
`OBSERVATION_INFORMATION` extension holds some ancillary information about the 
observations themselves. More on all of this below.

The `PRIMARY` header contains most of the information that isn't 
observation-specific. Here's what the primary header looks like for the 
calibration frames for our June 8, 2021 data. This should tell you everything
you might need to know or access about general observation setup. It also 
includes the units of the primary extension and a list of the reductions that
the pipeline applied to the data.

```
>>> print(repr(hdul['PRIMARY'].header))

SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -64 / array data type                                
NAXIS   =                    4 / number of array dimensions                     
NAXIS1  =                 4096 / number of spectral bins                        
NAXIS2  =                   20 / number of spatial bins                         
NAXIS3  =                   32 / number of echelle orders                       
NAXIS4  =                    5 / number of observations                         
EXTEND  =                    T                                                  
TARGET  = 'Jupiter '           / name of target body                            
BUNIT   = 'electrons/second'   / physical units of primary extension            
OBSERVER= 'de Kleer, Brown, Camarca, Milby' / last names of observers           
LAYOUT  = 'mosaic  '           / detector layout (legacy or mosaic)             
SLITLEN =                  7.0 / slit length [arcsec]                           
SLITWID =                1.722 / slit width [arcsec]                            
XDNAME  = 'red     '           / name of cross diserpser                        
XDANG   =                1.318 / cross disperser angle [deg]                    
ECHANG  =             -0.71594 / echelle angle [deg]                            
SPABIN  =                    3 / spatial binning [pix/bin]                      
SPEBIN  =                    1 / spectral binning [pix/bin]                     
SPASCALE=                0.358 / spatial bin scale [arcsec/bin]                 
SPESCALE=                0.179 / spectral bin scale [arcsec/bin]                
PIXWIDTH=                 15.0 / pixel width [micron]                           
REDUX00 = 'cosmic_rays_removed' / reduction applied to primary extension        
REDUX01 = 'bias_subtracted'    / reduction applied to primary extension         
REDUX02 = 'flat_field_corrected' / reduction applied to primary extension       
REDUX03 = 'gain_corrected'     / reduction applied to primary extension         
REDUX04 = 'rectified'          / reduction applied to primary extension         
REDUX05 = 'airmass_ext_corrected' / reduction applied to primary extension
```

Similar (but much less) informatino exists in the headers for the other 
extensions. For instance, the header for `BIN_EDGE_WAVELENGTHS` looks like 
this:

```
XTENSION= 'IMAGE   '           / Image extension                                
BITPIX  =                  -64 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                 4097 / number of spectral bin edges                   
NAXIS2  =                   32 / number of echelle orders                       
PCOUNT  =                    0 / number of parameters                           
GCOUNT  =                    1 / number of groups                               
EXTNAME = 'BIN_EDGE_WAVELENGTHS' / extension name                               
BUNIT   = 'nm      '           / wavelength physical unit
```

You can access observation-specific information from the table stored in the
`OBSERVATION_INFORMATION` extension. It includes the original filenames of each
of the observations, their exposure times, the start time of the observations
in ISOT format `YYYY-MM-DDTHH:MM:SS.sss`, and the airmass at the start of the
exposure.

```
>>> print(hdul['OBSERVATION_INFORMATION'].columns)
    
ColDefs(
    name = 'FILENAME'; format = '25A'
    name = 'EXPTIME'; format = 'D'; unit = 'seconds'
    name = 'OBSDATE'; format = '19A'
    name = 'AIRMASS'; format = 'E'
)
```

#### Science Target Data

The file structure of `science_observations.fits.gz` is similar, but has both
the target satellite data and the guide satellite data.

```
>>> from astropy.io import fits
>>> with fits.open('science_observations.fits.gz') as hdul:
>>>     hdul.info()

Filename: science_observations.fits.gz
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU      28   (4096, 20, 32, 17)   float64   
  1  GUIDE_SATELLITE    1 ImageHDU        12   (4096, 20, 32, 17)   float64   
  2  BIN_CENTER_WAVELENGTHS    1 ImageHDU         9   (4096, 32)   float64   
  3  BIN_EDGE_WAVELENGTHS    1 ImageHDU         9   (4097, 32)   float64   
  4  ECHELLE_ORDERS    1 BinTableHDU     11   32R x 1C   [I]   
  5  SCIENCE_OBSERVATION_INFORMATION    1 BinTableHDU     18   17R x 4C   [25A, D, 19A, E]   
  6  GUIDE_SATELLITE_OBSERVATION_INFORMATION    1 BinTableHDU     18   17R x 4C   [25A, D, 19A, E]   
```

For this file, the `PRIMARY` extension holds the target satellite data, and the
`GUIDE_SATELLITE` data holds the...wait for it...guide satellite data! It also
includes the same wavelength and echelle order information as the flux 
calibration data file, and has individual observation-specific information 
tables for the target and guide satellites.

### Retrieving Aurora Brightnesses

The function `get_aurora_brightnesses` takes the reduced data, flux-calibrates
the science images and extracts the surface brightness in rayleigh at the 
following wavelengths (if the detector setup captured them):
1. 557.7330 nm O(¹S) to O(¹D)
2. 630.0304 nm O(¹D) to O(³P₀)
3. 636.3776 nm O(¹D) to O(³P₂)
4. 656.2852 nm Balmer-alpha (Hα)
5. The 777.4 nm OI triplet:
   1. 777.1944 nm O(⁵P₃) to O(⁵S₂)
   2. 777.4166 nm O(⁵P₂) to O(⁵S₂)
   3. 777.5388 nm O(⁵P₁) to O(⁵S₂)
6. The 844.6 nm OI triplet
   1. 844.625 nm O(³P₀) to O(³S₁)
   2. 844.636 nm O(³P₂) to O(³S₁)
   3. 844.676 nm O(³P₁) to O(³S₁)

The brightness retrieval is slightly less automated than the reduction 
pipeline, so you may need to run it a few times and adjust some parameters 
until you achieve a good result. I haven't yet been able to figure out a way to
automate this stage, since it's more of an art than a science.

To run the retrieval:

```
>>> get_aurora_brightnesses(reduced_data_path=reduced_data_path, save_path=analysis_save_path, 
                            top_trim=2, bottom_trim=2, seeing=1.5, background_degree=1)

Retrieving brightnesses at 557.7 nm...
Retrieving brightnesses at 630.0 nm...
Retrieving brightnesses at 636.4 nm...
Retrieving brightnesses at 777.4 nm...
Retrieving brightnesses at 844.6 nm...
Retrieving brightnesses at 656.3 nm...
```

The first two arguments are simply paths to where you saved your reduced data
from the previous step (`reduced_data_path` is the same as the `save_path` 
used in `run_pipeline`), and the new `save_path` is the location where you want
the retrieval outputs saved. 

The arguments you may need to adjust are the following:
1. `top_trim` is how many rows to eliminate from the top of each order. The 
   rectification process will produce some weird, sawtooth-like effects and 
   they can mess up the fitting of the background and the estimation of the
   noise. A value of `top_trim=2` seems to be typical, but is is probably
   dependent on the binning.
2. `bottom_trim` is the same but for the bottom of each order. If the edge
   detection wasn't quite symmetric around the center of each order, the top
   and bottom trim values might not be the same. The default is `*_trim=0` for 
   both 1trim values. An initial run without any trimming might be useful to 
   get an idea of how many rows you should eliminate.
3. `seeing` accounts for the spread of the signal beyond the actual physical 
   size of the target satellite (in this case, the angular size). This argument
   lets you add to the radius of the target satellite to increase the size of 
   the aperture with which you capture the brightness. The aperture has a size
   $\pi(R+seeing)^2$ where $R$ is the apparent angular radius of the target
   satellite. I correct the final reported brightness by scaling it by the 
   ratio of the aperture size to the target size $((R+seeing)/R)^2$.
4. `background_degree` lets you choose the degree of the background polynomial
   fit. The default is `background_degree=1`, which I think is good in most 
   cases. Higher-degree polynomials can characterize noise which has more 
   spatial structure than a linear trend from one end of the slit to another,
   but you risk over-fitting and subtracting real brightness from the target
   satellite. I think in most cases it's probably better to keep it to a first-
   degree polynomial and more carefully estimate additional background in your
   brightness values.

You can evaluate the results of a retrieval by looking at the two output types.
The algorithm creates a `.txt` file which contains the observation start time,
the retrieved brightness in rayleigh and its estimated uncertainty. Here's an
example of one of those files for the 630.0 nm brightnesses from June 8, 2021:

```
2021-06-08T12:58:16 266.30 16.54
2021-06-08T13:05:29 222.44 15.15
2021-06-08T13:22:12 267.22 16.57
2021-06-08T13:29:41 279.32 16.93
2021-06-08T13:36:46 275.59 16.82
2021-06-08T13:44:33 305.46 17.68
2021-06-08T13:51:41 312.19 17.87
2021-06-08T13:58:52 244.22 15.85
2021-06-08T14:08:09 233.35 15.52
2021-06-08T14:15:28 248.32 16.00
2021-06-08T14:22:45 287.77 17.19
2021-06-08T14:29:56 338.80 18.62
2021-06-08T14:37:35 363.98 19.29
2021-06-08T14:44:47 320.27 18.12
2021-06-08T14:52:33 257.05 16.29
2021-06-08T15:01:18 223.69 15.26
2021-06-08T15:08:29 212.36 14.95
```

It also creates a directory into which it saves graphics for each of the 
observations, showing
1. The initial reduced data image,
2. The fitted background,
3. The background-subtracted image,
4. The isolated bins used for the target brightness, and
5. The isolated bins used for noise estimation.

That's pretty much all there is to it. Each time I retrieve brightnesses, I run
it a few times until I've figured out the right trim and seeing values, then 
use the results in the `.txt` file.

### Ancillary Functions
There are a few ancillary functions I've made available, mostly because I 
wanted an easy way to plot some of the data used in the calibrations. The
functions available are:
1. `get_meridian_reflectivity` which returns a dictionary containing 
   wavelengths and Jupiter meridian reflectivity (I/F) values from Woodman et
   al. (1979).[^5]
2. `get_mauna_kea_summit_extinction` which returns a dictionary containing
   wavelengths and airmass- and wavelength-dependent extinction for the summit
   of Mauna Kea from Buton et al. (2003).[^6]
3. `get_solar_spectral_radiance`: which returns a dictionary containing the 
   theoretical solar spectral radiance above Earth's atmosphere at 1 au from 
   the Sun. The original data were actually spectral irradiance (W/m²/nm) which 
   I converted to radiance by dividing by pi, giving units of W/m²/nm/sr. The 
   spectral irradiance data come from the 2000 ASTM Standard Extraterrestrial 
   Spectrum Reference E-490-00.[^7]

[^5]: Woodman, J. H., et al. (1979), Spatially resolved reflectivities of 
      Jupiter during the 1976 opposition, Icarus, 37(1), 73-83, 
      doi:10.1016/0019-1035(79)90116-7
[^6]: Buton, C., et al. (2012), Atmospheric extinction properties above Mauna 
      Kea from the nearby Supernova Factory spectro-photometric data set, 
      Astronomy & Astrophysics, 549, doi:10.1051/0004-6361/201219834
[^7]: https://www.nrel.gov/grid/solar-resource/spectra-astm-e490.html

