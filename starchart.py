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
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy.io.votable import parse_single_table
from warnings import warn

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
        self.Beelitz = EarthLocation(lat=52.239976 * u.deg, lon=12.964697 * u.deg)
        #TODO fix stupid summertime
        utcoffset = 1 * u.hour  # Eastern Daylight Time
        self.time = Time('2019-12-01 00:00:00') - utcoffset
        self.local_time = self.time + utcoffset
        altaz = self.stars.transform_to(AltAz(obstime=self.time, location=self.Potsdam))
        i = np.where(altaz.alt.deg > 0.0)
        self.az = altaz.az[i].deg
        self.alt = altaz.alt[i].deg
        self.magnitudes = self.magnitudes[i]
        self.names = self.names[i]
        self.theta = np.deg2rad(self.az)
        self.r = 90.0 - self.alt

    def _bodies(self):
        bodies = {'moon': '☽',
                  'mercury': '☿',
                  'venus': '♀',
                  'mars': '♂',
                  'jupiter': '♃',
                  'saturn': '♄'}
        for body_name in bodies:
            body = get_body(body_name, self.time, location=self.Potsdam)

            body_altaz = body.transform_to(AltAz(obstime=self.time, location=self.Potsdam))
            theta = body_altaz.az.rad
            r = 90.0 - body_altaz.alt.deg
            if body_altaz.alt> 0.0:
                plt.text(theta, r, bodies[body_name], color='darkred',
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=8)

    def _findstar(self, starname):
        for i, name in enumerate(self.names):
            if starname in name:
                print(i, name)
                return i
        print('Cannot find %s in list' % starname)
        return -1

    def _annotate(self, axis):
        names = [(b'3Alp Lyr', "Vega"),
                 (b'9Alp CMa', "Sirius"),
                 (b'13Alp Aur', "Capella"),
                 (b'10Alp CMi', "Prokyon"),
                 (b'58Alp Ori', "Betageuze"),
                 (b'Alp Tau', "Aldebaran"),
                 (b'19Bet Ori', "Rigel"),
                 (b'16Alp Boo', "Arktur"),
                 (b'53Alp Aql', "Altair")]
        for name in names:
            try:
                i = list(self.names).index(name[0])
            except ValueError:
                i = self._findstar(name[0])
            if i >= 0:
                axis.annotate(" "+name[1], (self.theta[i], self.r[i]), size=6, color='blue')

    def zodiac(self):
        zodiacs = {"Leo": ['10 41 11', '17 29 52', '♌'],
                   "Can": ['08 40 22.20', '19 40 19.00', '♋'],
                   "Vir": ['13 09 47.14', '-2 04 10.80', '♍'],
                   "Gem": ['06 56 57.92', '22 57 54.14', '♊'],
                   "Tau": ['04 47 11.90', '20 55 25.00', '♉'],
                   "Ari": ['02 34 00.00', '20 58 36.00', '♈'],
                   "Pis": ['00 37 39.80', '10 21 28.00', '♓']}
        for key in zodiacs:
            ra = zodiacs[key][0]
            de = zodiacs[key][1]
            sign = zodiacs[key][2]
            coord = SkyCoord(ra=ra, dec=de, unit=(u.hourangle, u.deg)).transform_to(AltAz(obstime=self.time, location=self.Potsdam))
            if coord.alt.deg > 0.0:
                plt.text(coord.az.rad, 90.0 - coord.alt.deg, sign, color='darkgreen',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=6)

    def _constellation(self, axis):
        from constellations import constellations

        for constellation in constellations:
            thetas = []
            rs = []
            names=[]

            while len(constellation) >= 1:
                try:
                    i0 = list(self.names).index(constellation[0])
                except ValueError:
                    i0 = self._findstar(constellation[0])
                if i0 >= 0:
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
        self._bodies()

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_thetagrids((0.0, 90.0, 180.0, 270.0), labels=('N', 'O', 'S', 'W'))
        self._constellation(ax)
        self.zodiac()
        ax.set_rmax(90.0)
        ax.set_rmin(0.0)
        ax.set_rgrids((10.0, 30.0, 50.0, 70.0), labels='')
        self._annotate(ax)

        plt.text(1.05, -0.07, 'Sternkarte für Potsdam %s Uhr, 2019 © Dr. Jörg Weingrill für die Sternfreunde Beelitz e.V.' %
                 self.local_time.strftime("%d.%m.%Y %H:%M"),
                 horizontalalignment='right', verticalalignment='bottom',
                 fontsize=6, transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig('/home/jwe/Pictures/Figures/starchart.pdf', dpi=300)
        plt.close()
        pass


if __name__ == "__main__":
    sc = StarChart()
    sc.plot()
