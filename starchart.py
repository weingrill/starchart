#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Joerg Weingrill"
__copyright__ = "Copyright 2019 Jörg Weingrill"
__credits__ = ["Joerg Weingrill"]
__license__ = "GPL"
__version__ = "0.1.1"
__maintainer__ = "Joerg Weingrill"
__email__ = "jweingrill@aip.de"
__status__ = "Development"
__date__ = "11/6/19"

import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.io.votable import parse_single_table


def scaleto(values, bounds, k=None, d=None):
    """
    scales values within bounds
    """
    x1, x2 = np.min(values), np.max(values)
    y1, y2 = bounds[0], bounds[1]
    if k is None:
        k = (y2 - y1) / (x2 - x1)
    if d is None:
        d = y1 - k * x1
    return k * values + d


class StarChart(object):
    def __init__(self):
        self._import_catalogue()
        self._coordinatetransform()
        pass

    def _import_catalogue(self):
        filename = '/home/jwe/data/BrightStarCatalogue.vot'
        votable = parse_single_table(filename)
        self.table = votable.to_table()
        i = np.where(self.table['Vmag'] < 7.0)
        self.table = self.table[i]
        stars_array = votable.array[i]

        self.stars = SkyCoord(stars_array['_RAJ2000'] * u.deg, stars_array['_DEJ2000'] * u.deg, frame='icrs')
        self.magnitudes = stars_array['Vmag']
        self.names = stars_array['Name']
        # fixing duplicate names
        self.names[stars_array['recno'] == 5054] = '79Zet1UMa'
        self.names[stars_array['recno'] == 5055] = '79Zet2UMa'
        self.names[stars_array['recno'] == 1948] = '50Zet1Ori'
        self.names[stars_array['recno'] == 1949] = '50Zet2Ori'
        self.names[stars_array['recno'] == 1851] = '34Del1Ori'
        self.names[stars_array['recno'] == 1852] = '34Del2Ori'

    def _coordinatetransform(self):
        self.Potsdam = EarthLocation(lat=52.4 * u.deg, lon=13.10 * u.deg, height=36 * u.m)
        utcoffset = 1 * u.hour  # Eastern Daylight Time
        self.time = Time('2019-12-01 23:00:00') - utcoffset
        altaz = self.stars.transform_to(AltAz(obstime=self.time, location=self.Potsdam))
        i = np.where(altaz.alt.deg > 0.0)
        self.az = altaz.az[i].deg
        self.alt = altaz.alt[i].deg
        self.magnitudes = self.magnitudes[i]
        self.names = self.names[i]
        self.theta = np.deg2rad(self.az)
        self.r = 90.0 - self.alt

    def _annotate(self, axis):
        names = [(b'3Alp Lyr', "Vega"),
                 (b'9Alp CMa', "Sirius"),
                 (b'13Alp Aur', "Capella"),
                 (b'10Alp Cmi', "Procyon"),
                 (b'58Alp Ori', "Betageuze"),
                 (b'Alp Tau', "Aldebaran"),
                 (b'19Bet Ori', "Rigel")]
        for name in names:
            i = np.where(self.names == name[0])[0]
            if i.size > 0:
                axis.annotate(" "+name[1], (self.theta[i], self.r[i]), size=6, color='blue')

    def _constellation(self, axis):
        def findstar(starname):
            for i, name in enumerate(self.names):
                if starname in name:
                    print(i, name)
                    return i
            raise ValueError(starname)

        constellations = ([b'10Alp CMi', b'3Bet CMi'],
                          [b'2Alp Tri', b'4Bet Tri', b'9Gam Tri', b'2Alp Tri'],
                          [b'45Eps Cas', b'37Del Cas', b'27Gam Cas', b'18Alp Cas', b'11Bet Cas'],
                          [b'21Alp And', b'53Bet Peg', b'54Alp Peg', b'88Gam Peg', b'21Alp And'],
                          [b'50Alp Cyg', b'37Gam Cyg', b'21Eta Cyg', b'6Bet1Cyg'],
                          [b'18Del Cyg', b'37Gam Cyg', b'53Eps Cyg'],
                          [b'50Alp UMa', b'48Bet UMa', b'64Gam UMa', b'69Del UMa', b'50Alp UMa'],
                          [b'69Del UMa', b'77Eps UMa', b'79Zet1UMa', b'85Eta UMa'],
                          [b'53Kap Ori', b'50Zet1Ori', b'58Alp Ori', b'37Phi1Ori', b'24Gam Ori', b'34Del1Ori', b'19Bet Ori'],
                          [b'112Bet Tau', b'37The Aur', b'34Bet Aur', b'13Alp Aur', b'10Eta Aur', b'3Iot Aur', b'112Bet Tau']
                          )

        for constellation in constellations:
            thetas = []
            rs = []
            names=[]
            while len(constellation) >= 1:
                i0 = np.where(self.names == constellation[0])[0]
                if len(i0) == 0:
                    i0 = findstar(constellation[0])
                if len(i0) > 1:
                    raise ValueError(constellation[0])
                thetas.append(self.theta[i0])
                rs.append(self.r[i0])
                names.append(self.names[i0])
                constellation = constellation[1:]
            try:
                axis.plot(thetas, rs, color='darkblue', linewidth=0.5)
            except ValueError:
                print(names)

    def plot(self):
        i = np.where(self.magnitudes < 4.5)
        k = np.where((self.magnitudes >= 4.5) & (self.magnitudes < 6.0))
        area = 0.5 * scaleto(self.magnitudes[i], [5, 0.5])**2
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
#        ax.plot(self.theta[k], self.r[k], ',', color='lightgray', markersize=0.5)
        ax.scatter(self.theta[i], self.r[i], s=area, color='k', )  # cmap='coolwarm_r'
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_thetagrids((0.0, 90.0, 180.0, 270.0), labels=('N', 'E', 'S', 'W'))
        self._constellation(ax)
        ax.set_rmax(90.0)
        ax.set_rmin(0.0)
        ax.set_rgrids((10.0, 30.0, 50.0, 70.0), labels='')
        self._annotate(ax)

        plt.text(1.05, -0.07, 'Sternkarte für Potsdam %s, 2019 © Dr. Jörg Weingrill für die Sternfreunde Beelitz e.V.' % self.time,
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=6, transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('/home/jwe/Pictures/Figures/starchart.pdf', dpi=300)
        plt.close()
        pass


if __name__ == "__main__":
    sc = StarChart()
    sc.plot()
