import numpy as np
import astropy.io.fits
import sys
import os
import pickle
import astropy.units as u
from astropy.constants import h, c
#import pysynphot as S 
#from spectres import spectres 
import pymsg



def _centers_to_edges(centers):
    """Convert monotonic wavelength bin centers to bin edges."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("Need at least two wavelength centers.")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def _norm_axis_label(label):
    return "".join(ch for ch in label.lower() if ch.isalnum())


def _make_msg_params(specgrid, temperature, feh, logg):
    x = {}
    for axis_label in specgrid.axis_labels:
        key = _norm_axis_label(axis_label)
        if key in ("teff", "temperature", "temp", "effectivetemperature"):
            x[axis_label] = float(temperature)
        elif key in ("logg", "surfacegravity", "log10g"):
            x[axis_label] = float(logg)
        elif key in ("feh", "mh", "metallicity", "met", "m"):
            x[axis_label] = float(feh)
        else:
            raise KeyError(
                f"Unsupported MSG axis '{axis_label}'. "
                f"Expected something mappable to Teff/logg/[Fe/H]. "
                f"Grid axes are: {specgrid.axis_labels}"
            )
    return x

def create_stellar_grid_data(logg, feh, startag):
    def air_to_vac(wavelength):
        """
        Implements the air to vacuum wavelength conversion described in eqn 65 of
        Griesen 2006. Taken from specutils.
        """
        wlum = wavelength.to(u.um).value
        return (1 + 1e-6*(287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4)) * wavelength

    binned_wavelengths = np.load("../platon/data/low_res_lambdas.npy")  # meters (bin centers)
    # clip to 5.5 microns (currently 30 )
    binned_wavelengths = binned_wavelengths[binned_wavelengths < (5.5 * 1e-6)]

    lam_edges_angstrom = _centers_to_edges(binned_wavelengths) * 1e10  # MSG expects Angstrom edges

    
    output_spectra = {}
    temps = []
    spectra = []

    MSG_DIR = os.environ['MSG_DIR']
    GRID_DIR = os.path.join(MSG_DIR, 'data', 'grids')

    specgrid_file_name = os.path.join(GRID_DIR, 'sg-Goettingen-HiRes.h5')

    specgrid = pymsg.SpecGrid(specgrid_file_name)


    for temperature in np.arange(3500, 12000, 100):
        #filename = "bt-settl-agss/lte{0:03d}-4.5-0.0a+0.0.BT-Settl.7.dat.txt".format(int(temperature/100))
        #alt_filename = "bt-settl-agss/lte0{0}-4.5-0.0.BT-Settl.7.dat.txt".format(int(temperature/100))
        
        #if os.path.isfile(filename):
        #    wavelengths, spectrum = np.loadtxt(filename, unpack=True)
        #elif os.path.isfile(alt_filename):
        #    wavelengths, spectrum = np.loadtxt(alt_filename, unpack=True)
        #else:
        #    continue
        
        # MSG equivalent of the old Icat path:
        #   sp.flux [erg/s/cm^2/A] -> W/m^2/m, then divide by pi to get intensity-like quantity.
        x = _make_msg_params(specgrid, temperature, feh, logg)
        flux_grid = specgrid.flux(x=x, z=0.0, lam=lam_edges_angstrom, order=3)
        binned_spectrum = (flux_grid * 1e-7 * 1e4 * 1e10) / np.pi


        '''
        wavelengths *= u.Angstrom
        wavelengths = air_to_vac(wavelengths)
        spectrum *= (u.erg/u.cm**2/u.s/u.Angstrom)

        binned_spectrum = sp.binwave
        
        binned_spectrum = []


        avg_log_interval = np.median(np.diff(np.log10(np.unique(binned_wavelengths).value)))
        conversion_factor = None

        for i, wavelength in enumerate(binned_wavelengths):
            start = wavelength * 10**(-avg_log_interval/2.0)
            end = wavelength * 10**(avg_log_interval/2.0)

            cond = np.logical_and(wavelengths >= start, wavelengths < end)
            flux = np.mean(spectrum[cond])

            if conversion_factor is None:
                conversion_factor = flux.si.value / flux.value

            binned_spectrum.append(flux.value)
        #print(len(binned_spectrum))
        binned_spectrum = np.array(binned_spectrum) * conversion_factor
        '''



        spectra.append(binned_spectrum)
        temps.append(temperature)
        print(temperature, np.min(binned_spectrum), np.max(binned_spectrum))

    output_spectra['temperatures'] = np.array(temps)
    output_spectra['spectra'] = np.array(spectra)
    output_spectra['wavelengths_m'] = np.array(binned_wavelengths)

    with open(f"/project/ekempton/tfairnington/retrievals/platon/platon/data/{startag}", "wb") as f:
        pickle.dump(output_spectra, f)

        
    return 
