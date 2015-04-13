#!/usr/bin/env python
from distutils.core import setup,Extension
import sys


setup(
  name='xfab',
  version='0.0.2',
  description='Crystallographic toolbox and library',
  license='GPL', maintainer='Henning Osholm Soerensen and Jon Wright',
  maintainer_email='henning.sorensen@risoe.dk or wright@esrf.eu',
  download_url='http://sourceforge.net/project/showfiles.php?group_id=82044&package_id=309377',
  url='http://fable.wiki.sourceforge.net/xfab',
  packages=['xfab'],
  package_dir={"xfab": "xfab"},
  scripts=["scripts/gff_to_ubi.py",
           "scripts/ubi_to_gff.py",
           "scripts/plot_gff.py",
           "scripts/make_gve.py",
           "scripts/findpeaks",
           "scripts/grainspotter_loop",
           "scripts/makemap_all",
           "scripts/tweakdetpars",
           "scripts/tweakpars_FF.py",
           "scripts/tweakpars_NF.py"]
)
