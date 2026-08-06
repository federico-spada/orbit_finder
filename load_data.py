import numpy as np
import spiceypy as spice
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from astropy import units as u
from astropy_healpix import HEALPix
from dataclasses import dataclass, replace

import constants as cs

@dataclass
class ObservationData:
    # basic
    et: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    Rs: np.ndarray
    # uncertainty
    rmsRAs: np.ndarray
    rmsDec: np.ndarray
    rmsCorr: np.ndarray
    seeing: np.ndarray
    logSNR: np.ndarray
    # metadata
    notes: np.ndarray
    utcDate: np.ndarray
    prog: np.ndarray
    mode: np.ndarray
    astCat: np.ndarray
    astCatCode: np.ndarray
    stn: np.ndarray

    @property
    def numObs(self):
        return len(self.et)


def LoadDataADES(filename, mpco_file, ccod_file, 
    start_date=None, 
    end_date=None, 
    defaultSigma=2.0):
    # load file with obs. stations data
    with open(mpco_file, 'r') as f:
        obs_data = json.load(f)
    # load file with astrometric catalog codes
    with open(ccod_file, 'r') as f:
        cco_data = json.load(f)
    # load file with observations (.xml format)
    tree = ET.parse(filename)
    root = tree.getroot()
    fields = ["stn", "obsTime", "ra", "dec", "rmsRA", "rmsDec", "pos1", "pos2", "pos3",
              "prog", "mode", "ctr", "sys", "astCat", "rmsCorr", "seeing", "logSNR", "notes"]
    data = {field: [] for field in fields}
    for optical in root.findall("optical"):
        row = {child.tag: child.text for child in optical}
        for field in fields:
            data[field].append(row.get(field))
    stn     = np.array(data['stn'])
    et      = np.array([spice.str2et(ot) for ot in data['obsTime']])
    ra      = np.radians(np.array(data['ra'], dtype=float))
    dec     = np.radians(np.array(data['dec'], dtype=float))
    rmsRAs  = np.array(data['rmsRA'], dtype=float) ### includes the cos(Dec) factor!
    rmsDec  = np.array(data['rmsDec'], dtype=float)
    rmsCorr = np.array(data['rmsCorr'], dtype=float)
    pos1    = np.array(data['pos1'], dtype=float)
    pos2    = np.array(data['pos2'], dtype=float)
    pos3    = np.array(data['pos3'], dtype=float)
    prog    = np.array(data['prog'], dtype=str)
    mode    = np.array(data['mode'])
    ctr     = np.array(data['ctr'])
    sys     = np.array(data['sys'])
    seeing  = np.array(data['seeing'], dtype=float)
    logSNR  = np.array(data['logSNR'], dtype=float)
    notes   = np.array(data['notes'], dtype=str)
    astCat  = np.array(data['astCat'])
    utcDate = np.array([datetime.fromisoformat(spice.et2utc(eti, 'ISOC', 7)) for eti in et])
    m = len(et)
    Rs = np.zeros((m, 3))
    for i in range(m):
        obsloc = obs_data[stn[i]]
        if ctr[i]:
            r_ctr = spice.spkpos(str(int(ctr[i])), et[i], 'J2000', 'NONE', '10')[0]
        else:
            r_ctr = spice.spkpos('399', et[i], 'J2000', 'NONE', '10')[0]
        if 'Longitude' in obsloc:
            long = np.radians(obsloc['Longitude'])
            rcos = obsloc['cos'] * cs.RE
            rsin = obsloc['sin'] * cs.RE
            R0 = spice.cylrec(rcos, long, rsin)
            U = spice.pxform('ITRF93', 'J2000', et[i])
        else:
            if stn[i] == '247':
                lon = np.radians(float(data['pos1'][i]))
                lat = np.radians(float(data['pos2'][i]))
                alt = float(data['pos3'][i]) / 1e3 # m to km
                R0 = spice.georec(lon, lat, alt, cs.RE, cs.FE)
            else:
                R0 = np.array([pos1[i], pos2[i], pos3[i]])
            if sys[i] == 'WGS84':
                U = spice.pxform('ITRF93', 'J2000', et[i])
            elif sys[i] == 'ICRF_KM':
                U = np.eye(3)
            elif sys[i] == 'ICRF_AU':
                U = np.eye(3)
                R0 *= cs.AU
        Rs[i] = U @ R0 + r_ctr
    # note: units of uncertainties are arc sec!
    n_default = np.isnan(rmsDec).sum()
    rmsRAs = np.where(~np.isnan(rmsRAs), rmsRAs, defaultSigma * np.cos(dec))
    rmsDec = np.where(~np.isnan(rmsDec), rmsDec, defaultSigma)
    rmsCorr = np.where(~np.isnan(rmsCorr), rmsCorr, 0.0)
    # add catalog codes
    astCatCode = np.array([cco_data.get(cc, {}).get('Code', ' ') for cc in astCat])

    # optionally, use only observations between start_date and end_date
    if not start_date:
        start_date = spice.et2utc(et[0], 'ISOC', 2)
        i_start = 0
    else:
        et_start = spice.str2et(start_date)
        i_start = np.searchsorted(et, et_start, side='left')
    if not end_date:
        end_date = spice.et2utc(et[-1], 'ISOC', 2)
        i_end = len(et)
    else:
        et_end = spice.str2et(end_date)
        i_end = np.searchsorted(et, et_end, side='right')
    mask = slice(i_start, i_end)
    # unit conversion
    et /= cs.DAYS
    Rs /= cs.AU
    return ObservationData(
        et=et[mask],
        ra=ra[mask],
        dec=dec[mask],
        Rs=Rs[mask],
        rmsRAs=rmsRAs[mask],
        rmsDec=rmsDec[mask],
        rmsCorr=rmsCorr[mask],
        seeing=seeing[mask],
        logSNR=logSNR[mask],
        notes=notes[mask],
        utcDate=utcDate[mask],
        prog=prog[mask],
        mode=mode[mask],
        astCat=astCat[mask],
        astCatCode=astCatCode[mask],
        stn=stn[mask],
    ), n_default



def DebiasData(Data, bias_file):
    # read bias file >>>
    with open(bias_file, 'r') as file:
        lines = file.readlines()[:5]
    nside = int(lines[1][9:11])
    hp = HEALPix(nside=nside)
    catalogs = lines[4][1:].strip().split()
    bias = np.loadtxt(bias_file, skiprows=23)
    # <<<
    cid = [catalogs.index(cc) if cc in catalogs else -1 for cc in Data.astCatCode]
    pid = hp.lonlat_to_healpix(Data.ra * u.rad, Data.dec * u.rad)
    dRA, dDE, pmRA, pmDE = np.array( [bias[pid[i], 4*cid[i]:4*cid[i]+4] if cid[i] != -1
                                  else [0, 0, 0, 0] for i in range(Data.numObs)] ).T
    count = sum([c >= 0 for c in cid])
    dt = Data.et/365.25
    delta_ra = (dRA + dt * pmRA/1e3) / np.cos(Data.dec)
    delta_de =  dDE + dt * pmDE/1e3
    new_ra  = Data.ra  - delta_ra / cs.ARCSEC
    new_dec = Data.dec - delta_de / cs.ARCSEC
    return replace(Data, ra=new_ra, dec=new_dec), count



def AssignUncertaintiesVeres17(Data, defaultSigma=2.0, set_all=False):
    m = Data.numObs
    sigma = np.repeat(defaultSigma, m)
    ###
    for i in range(m):
        utcd = Data.utcDate[i]
        prog = Data.prog[i]
        mode = Data.mode[i]
        acat = Data.astCat[i]
        obsc = Data.stn[i]
        ### modern observations: CCD, CMO
        if mode in ['CCD', 'CMO', 'VID']:
            # generic w/ or w/o catalog information
            if acat != ' ':
                sigma[i] = 1.0
            else:
                sigma[i] = 1.5

            # Table 2: Catalina, Spacewatch, NEAT >>>              
            if obsc == '703':
                if utcd < datetime(2014, 1, 1):
                    sigma[i] = 1.0
                else:
                    sigma[i] = 0.8
            if obsc == '691':
                if utcd < datetime(2003, 1, 1):
                    sigma[i] = 0.6
                else:
                    sigma[i] = 0.5
            if obsc == '644':
                if utcd < datetime(2003, 9, 1):
                    sigma[i] = 0.6
                else:
                    sigma[i] = 0.4
            # <<<
            # Table 3: selected CCD surveys, and generic CCD >>>
            elif obsc == '704':
                sigma[i] = 1.0
            elif obsc == 'G96':
                sigma[i] = 0.5
            elif obsc == 'F51':
                sigma[i] = 0.2
            elif obsc == 'G45':
                sigma[i] = 0.6
            elif obsc == '699':
                sigma[i] = 0.8
            elif obsc == 'D29':
                sigma[i] = 0.75
            elif obsc == 'C51':
                sigma[i] = 1.0
            elif obsc == 'E12':
                sigma[i] = 0.75
            elif obsc == '608':
                sigma[i] = 0.6
            elif obsc == 'J75':
                sigma[i] = 1.0
            # <<<
            # Table 4: NEO follow-up observers >>>
            elif obsc == '645' and acat != ' ':
                sigma[i] = 0.3
            elif obsc == '673' and acat != ' ':
                sigma[i] = 0.3
            elif obsc == '689' and acat != ' ':
                sigma[i] = 0.5
            elif obsc == '950' and acat != ' ':
                sigma[i] = 0.5
            elif obsc == 'H01' and acat != ' ':
                sigma[i] = 0.3
            elif obsc == 'J04' and acat != ' ':
                sigma[i] = 0.4
            elif obsc == 'W84' and acat != ' ':
                sigma[i] = 0.5
            # Mt. Graham-LBT, observations by M. Micheli
            elif obsc == 'G83' and (prog == '02'):
                if acat == 'UCAC4' or acat == 'PPMXL':
                    sigma[i] = 0.3
                elif acat.startswith('Gaia'):
                    sigma[i] = 0.2
            # Las Cumbres Observatories
            elif obsc in ['K92', 'K93', 'Q63', 'Q64', 'V37', 'W85',
                'W86', 'W87', 'K91', 'E10', 'F65'] and acat != ' ':
                sigma[i] = 0.4
            elif obsc == 'Y28' and (acat == 'PPMXL' or acat.startswith('Gaia')):
                sigma[i] = 0.3
            elif obsc == '568':
                if acat == 'USNOB1' or acat == 'USNOB2':
                    sigma[i] = 0.5
                elif acat.startswith('Gaia'):
                    sigma[i] = 0.1
                elif acat == 'PPMXL':
                    sigma[i] = 0.2
            elif obsc == 'T09' and acat.startswith('Gaia'):
                sigma[i] = 0.1
            elif obsc == 'T12' and acat.startswith('Gaia'):
                sigma[i] = 0.1
            elif obsc == 'T14' and acat.startswith('Gaia'):
                sigma[i] = 0.1
            # Cerro Paranal, observations by M. Micheli  
            elif obsc == '309' and (prog == '0F'):
                if acat == 'UCAC04' or acat == 'PPMXL':
                    sigma[i] = 0.3
                elif acat.startswith('Gaia'):
                    sigma[i] = 0.2
            # <<<  
            # Space-based observations:
            # HST
            elif obsc == '250':
                sigma[i] = 0.25
            # Wise
            elif obsc == 'C51':
                sigma[i] = 2.5
            # JWST
            elif obsc == '274':
                sigma[i] = 0.05
            # Hipparcos (cf. Table 5)
            elif obsc == '248':
                sigma[i] = 0.2
            # Lucy
            elif obsc == '336':
                 sigma[i] = 2.0
            # Psyche
            elif obsc == '338':
                 sigma[i] = 2.0
            # Trace Gas Orbiter
            elif obsc == '339':
                 sigma[i] = 4.0
            # TESS
            elif obsc == 'C57':
                 sigma[i] = 20.0
        # Table 5: non-CCD observations >>>
        elif mode == 'PHO':
            if utcd > datetime(1950, 1, 1):
                sigma[i] = 2.5
            elif utcd > datetime(1890, 1, 1):
                sigma[i] = 5.0
            else:
                sigma[i] = 10.
        elif mode == 'OCC':
            sigma[i] = 0.2
        elif mode == 'MER':
            sigma[i] = 0.5
        elif mode == 'ENC':
            sigma[i] = 0.75
        elif mode == 'MIC':
            sigma[i] = 2.0
       # <<<     
    # note: units of uncertainties are arc sec!
    rmsRAs = sigma * np.cos(Data.dec)
    rmsDec = sigma
    if set_all:
        ### set all uncertainties according to Veres+17 model:
        new_rmsRAs  = rmsRAs
        new_rmsDec  = rmsDec
        new_rmsCorr = np.zeros_like(rmsRAs)
    else:
        ### set uncertainties according to Veres+17 model only when not provided:
        new_rmsRAs  = Data.rmsRAs                                                    
        new_rmsDec  = Data.rmsDec                                          
        new_rmsCorr = Data.rmsCorr
        valid = ~(np.isnan(new_rmsRAs) | np.isnan(new_rmsDec))
        new_rmsRAs  = np.where(valid, new_rmsRAs, rmsRAs)
        new_rmsDec  = np.where(valid, new_rmsDec, rmsDec)
        new_rmsCorr = np.where(valid, new_rmsCorr, 0.0)
    return replace(Data, rmsRAs=new_rmsRAs, rmsDec=new_rmsDec, rmsCorr=new_rmsCorr)


if __name__ == "__main__":


    spice.furnsh('_aux/spice_orbit_finder.mkn')

    mpco_file = '_aux/obscodes_extended.json'
    ccod_file = '_aux/AstCatWithCodes.json'

    DataRaw, n1 = LoadDataADES('153P.xml', mpco_file, ccod_file)
    Data, n2 = DebiasData(DataRaw, '_aux/bias.dat')

    # to use Veres+17 set defaultSigma to np.nan when loading MPC file
    #DataRawV, n1V = LoadDataADES('153P.xml', mpco_file, ccod_file, defaultSigma=np.nan)
    #DataDebiasedV, n2V = DebiasData(DataRawV, '_aux/bias.dat')
    #DataV = AssignUncertaintiesVeres17(DataRawV)


    spice.kclear() 

