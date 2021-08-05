#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function
from .due import due, Doi
from .download_data import download_Algonauts2021

__all__ = []


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi(""),
         description="Buzznauts: NMA-DL 2021 elated-buzzwards pod project",
         tags=["NMA-elated buzzards-Buzznauts-project"],
         path='Buzznauts')
