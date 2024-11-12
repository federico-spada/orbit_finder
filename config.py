### constants 
RE = 6378.1366 # km
FE = 1./298.257223563 ### Earth oblateness, WGS84 value 
AU   = 1.495978707e8 # km
CC = 299792.458 # km/s  
DAYS = 86400 # s 
MU_S = 132712440041.279419 # km^3/s^2


### parameters for propagation with SciPy ode solver:
# gravitational parameters of planets (DE440, km^3/s^2)
MU_P = [22031.868551, 324858.592000, 398600.435507, 42828.375816, \
        126712764.100000, 37940584.841800, 5794556.400000, \
        6836527.100580, 975.500000, 4902.800118]
# gravitational parameter of asteroids (DE440, km^3/s^2)
MU_A = [62.628889, 13.665878, 1.920571, 17.288233, 0.646878, 1.139872,\
        5.625148, 2.023021, 1.589658, 0.797801, 2.683036, 0.938106, \
        2.168232, 1.189808, 3.894483, 2.830410]
# names of planets in SPICE Kernel
TAG_P = ['1', '2', '399', '4', '5', '6', '7', '8', '9', '301']
# names of asteroids in SPICE Kernel
TAG_A = ['2000001', '2000002', '2000003', '2000004', '2000006', '2000007', \
       '2000010', '2000015', '2000016', '2000029', '2000052', '2000065', \
       '2000087', '2000088', '2000511', '2000704']
# rescale constants to use units of au, days:
MU_S = MU_S * DAYS**2/AU**3
MU_P = [ MU_P_i * (DAYS**2/AU**3) for MU_P_i in MU_P ]
MU_A = [ MU_A_i * (DAYS**2/AU**3) for MU_A_i in MU_A ]
CC = CC * DAYS/AU


### parameters for propagation with rebound/assist:
# large DE441 planets file:
assist_planets_file = '/Users/fs255/rebound_assist/data/linux_m13000p17000.441'
# alternatively, use smaller DE440 planets file (shorter timespan):
#assist_planets   = '/Users/fs255/rebound_assist/data/linux_p1550p2650.440'
# asteroids file
assist_asteroids_file = '/Users/fs255/rebound_assist/data/sb441-n16.bsp'


### R.A., Decl. uncertainties for a selection of observatories (Veres et al. 2017)
uncertainty = { '703': 1.0 , '691': 0.6 , '644': 0.6 ,
                '704': 1.0 , 'G96': 0.5 , 'F51': 0.2 , 'G45': 0.6 , '699': 0.8 ,
                'D29': 0.75, 'C51': 1.0 , 'E12': 0.75, '608': 0.6 , 'J75': 1.0 ,
                '645': 0.3 , '673': 0.3 , '689': 0.5 , '950': 0.5 , 'H01': 0.3 ,
                'J04': 0.4 , 'G83': 0.3 , 'K92': 0.4 , 'K93': 0.4 , 'Q63': 0.4 ,
                'Q64': 0.4 , 'V37': 0.4 , 'W84': 0.4 , 'W85': 0.4 , 'W86': 0.4 ,
                'W87': 0.4 , 'K91': 0.4 , 'E10': 0.4 , 'F65': 0.4 , 'Y28': 0.3 ,
                '568': 0.25, 'T09': 0.1 , 'T12': 0.1 , 'T14': 0.1 , '309': 0.3 ,
                '250': 0.05, 'C57': 0.05 } # arc sec


