# Copyright (c) 2016 Weitian LI <liweitianux@live.com>
# MIT license


"""
Retrieve the Galactic SNRs catalog data by parsing the web by /D. A. Green/:

    http://www.mrao.cam.ac.uk/surveys/snrs/
    http://www.mrao.cam.ac.uk/surveys/snrs/snrs.data.html
"""


import os
import re
import logging
from collections import OrderedDict

import requests
import bs4


logger = logging.getLogger(__name__)


class SNRDataGreen:
    """Class for Green's Galactic SNRs catalog data parse and manipulation.

    The available SNR data:
    - glon, glat : Galactic longitude, latitude (rounded to 0.1deg) [degree]
    - ra, dec : Right ascension, Declination (J2000) [degree]
    - size : angular size [degree]: (diameter, diameter) or (major, minor)
    - type : shape type: (shape, flag) with shapes of "S" (shell),
             "F" (filled-center), "C" (composite), and flag "?" if uncertain.
    - flux : flux density at 1 GHz [Jy]
    - specindex : Spectral index of the integrated radio emission
    - other_names : other name(s) commonly used for the SNR

    For more detailed description about the SNR catalog, refer to:
    http://www.mrao.cam.ac.uk/surveys/snrs/snrs.info.html
    """
    def __init__(self, dstr):
        self.data = self.parse(dstr)

    @classmethod
    def parse(cls, dstr):
        """Parse the SNR data string, for one SNR object.

        Parameters
        ----------
        dstr : str
            String containing the SNR data

        Returns
        -------
        data : dict
            A data dictionary containing the parsed SNR data
        """
        pattern = re.compile((
            r"^\s*(?P<glon>\d+\.\d+)\s+(?P<glat>[-+]?\d+\.\d+)\s+"
            r"(?P<ra>\d{2}\s+\d{2}\s+\d{2})\s+(?P<dec>[-+]?\d{2}\s+\d{2})\s+"
            r"(?P<size>[0-9.]+\??|[0-9.]+x[0-9.]+\??|\?)\s+"
            r"(?P<shape>[SCF?]{1,2})\s+"
            r"(?P<flux>\>?\d+\.\d+\??|\>?\d+\??|\?)\s+"
            r"(?P<specindex>\d+\.\d+\??|\d+\??|\?|varies)\s*"
            r"(?P<othernames>.*)$"))
        match = pattern.match(dstr)
        data = OrderedDict([
            ("glon", float(match.group("glon"))),
            ("glat", float(match.group("glat"))),
            ("ra", cls._parse_ra(match.group("ra"))),
            ("dec", cls._parse_dec(match.group("dec"))),
            ("size", cls._parse_size(match.group("size"))),
            ("shape", cls._parse_shape(match.group("shape"))),
            ("flux", cls._parse_flux(match.group("flux"))),
            ("specindex", cls._parse_specindex(match.group("specindex"))),
            ("othernames", cls._parse_othernames(match.group("othernames"))),
        ])
        return data

    @staticmethod
    def _parse_ra(s):
        """Parse the R.A. string "hh mm ss" to degree [0, 360)"""
        pattern = re.compile(r"(?P<hh>\d+)\s+(?P<mm>\d+)\s+(?P<ss>\d+)")
        match = pattern.match(s)
        hh = float(match.group("hh"))
        mm = float(match.group("mm"))
        ss = float(match.group("ss"))
        return (hh*15.0 + mm*15.0/60.0 + ss*15.0/3600.0)

    @staticmethod
    def _parse_dec(s):
        """Parse the Dec. string "dd mm" to degree [-90, 90]"""
        pattern = re.compile(r"(?P<sign>[-+]?)(?P<dd>\d+)\s+(?P<mm>\d+)")
        match = pattern.match(s)
        if match.group("sign") == "-":
            sign = -1.0
        else:
            sign = 1.0
        dd = float(match.group("dd"))
        mm = float(match.group("mm"))
        return sign * (dd + mm/60.0)

    @staticmethod
    def _parse_size(s):
        """Parse the SNR angular size string.

        Returns
        -------
        major : float
        minor : float
        flag : str
            (diameter, diameter) of the SNR if approximately circular;
            (major axis, minor axis) if SNR is elongated.
            All values are in unit [ degree ].
            Possible flag: "", "?" (uncertain)
        """
        if s.endswith("?"):
            flag = "?"
            s = s.rstrip("?")
        else:
            flag = ""
        try:
            major, minor = map(float, s.split("x"))
        except ValueError:
            major = minor = float(s)
        return (major, minor, flag)

    @staticmethod
    def _parse_shape(s):
        """Parse the SNR shape (a.k.a. type) string.

        Returns
        -------
        shape : str
        flag : str
            Possible shapes are "S" (shell), "F" (filled-center),
            "C" (composite), or None (very uncertain);
            Possible flag: "", "?" (uncertain)
        """
        flag = ""
        if s.endswith("?"):
            flag += "?"
            s = s.rstrip("?")
        if s != "":
            shape = s
        else:
            shape = None
        return (shape, flag)

    @staticmethod
    def _parse_flux(s):
        """Parse the flux density string.

        Returns
        -------
        flux : float
        flag : str
            Flux density [ Jy ] at 1GHz, None if the value is uncertain.
            Possible flag: "", "?", ">", ">?"
        """
        flag = ""
        if s.startswith(">"):
            flag += ">"
            s = s.lstrip(">")
        if s.endswith("?"):
            flag += "?"
            s = s.rstrip("?")
        try:
            flux = float(s)
        except ValueError:
            flux = None
        return (flux, flag)

    @staticmethod
    def _parse_specindex(s):
        """Parse the spectral index string.

        Returns
        -------
        specindex : float
        flag : str
            Spectral index, None if the value is uncertain.
            Possible flag: "", "?", "varies"
        """
        if s == "varies":
            specindex = None
            flag = "varies"
        elif s.endswith("?"):
            flag = "?"
            s = s.rstrip("?")
            try:
                specindex = float(s)
            except ValueError:
                specindex = None
        else:
            specindex = float(s)
            flag = ""
        return (specindex, flag)

    @staticmethod
    def _parse_othernames(s):
        """Parse the other names string to a list of names."""
        s = s.strip()
        if s:
            return s.split(",")
        else:
            return []

    @property
    def name(self):
        pattern = "G{glon:05.1f}{glat:+05.1f}"
        return pattern.format(**self.data)

    @property
    def othernames(self):
        return self.data["othernames"]

    @property
    def glon(self):
        return self.data["glon"]

    @property
    def glat(self):
        return self.data["glat"]

    @property
    def ra(self):
        return self.data["ra"]

    @property
    def dec(self):
        return self.data["dec"]

    @property
    def size(self):
        return self.data["size"]

    @property
    def shape(self):
        return self.data["shape"]

    @property
    def flux(self):
        return self.data["flux"]

    @property
    def specindex(self):
        return self.data["specindex"]

    @property
    def data_flat(self):
        """Get the data with tuple items flattened for easier CSV process"""
        data = OrderedDict([
            ("name", self.name),
            ("glon", self.glon),
            ("glat", self.glat),
            ("ra", self.ra),
            ("dec", self.dec),
            ("size_major", self.size[0]),
            ("size_minor", self.size[1]),
            ("size_flag", self.size[2]),
            ("shape", self.shape[0]),
            ("shape_flag", self.shape[1]),
            ("flux", self.flux[0]),
            ("flux_flag", self.flux[1]),
            ("specindex", self.specindex[0]),
            ("specindex_flag", self.specindex[1]),
            ("othernames", self.othernames),
        ])
        return data


def retrieve_snr_data_green(url):
    """Retrieve D. A. Green's Galactic SNRs catalog and parse the HTML
    contents to extract the catalog data.

    Parameters
    ----------
    url : str
        URL to the D. A. Green's SNRs catalog summary data page,
        can also be the path to the local HTML file.

    Returns
    -------
    snrdata : list[str]
        A string list with each line representing the information of
        one SNR object.

        Data string format:
        - Column 1, 2: Galactic longitude (l) and latitude (b)
        - Column 3-5: R.A. J2000 (hh mm ss)
        - Column 6, 7: Dec. J2000 (dd mm)
        - Column 8: Size [ arcmin ], `r` if circular, `Mxm` if elliptical;
                    may also contains a "?"
        - Column 9: Type (e.g., S, C, S?, C?)
        - Column 10: Flux density at 1 GHz [ Jy ]
        - Column 11: Spectral index (may contains "?" or be "varies")
        - Column 12: Other name(s), separated by ","
    """
    # Strip the beginning "file://" if presents
    url = re.sub(r"^file://", "", url)
    logger.info("Retrieve Galactic SNRs catalog from: {0}".format(url))
    if os.path.exists(url):
        # A local HTML file
        html = open(url).read()
    else:
        # Remote web page
        r = requests.get(url)
        r.raise_for_status()
        html = r.text
    logger.info("Parse the HTML contents ...")
    soup = bs4.BeautifulSoup(html, "html.parser")
    snrdata_tag = soup.body.pre
    snrdata_str = [ch.strip() if isinstance(ch, bs4.element.NavigableString)
                   else ch.string.strip()
                   for ch in snrdata_tag.children]
    # Remove the header row
    del snrdata_str[0]
    # Strip the last data row
    snrdata_str[-1] = re.sub(r"[-\s]*$", "", snrdata_str[-1])
    # The remaining SNR data string list should be even-length, since
    # every two items are the Galactic coordinate and other information.
    if len(snrdata_str) % 2 != 0:
        raise ValueError("length of the parsed SNR data str list is ODD")
    # Concatenate every two items corresponding to the same SNR object
    snrdata_str2 = [" ".join(x) for x in zip(snrdata_str[0::2],
                                             snrdata_str[1::2])]
    logger.info("Done parse the HTML contents: "
                "got {0} SNR objects".format(len(snrdata_str2)))
    return snrdata_str2
