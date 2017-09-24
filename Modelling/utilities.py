"""
A few utilities for processing and preparing source data with the
hazard modeller's toolkit and OpenQuake Engine
"""
import os
import shapefile
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from openquake.baselib.node import Node
from openquake.hazardlib import nrml
from openquake.hazardlib.geo import (Point, Line, Polygon, NodalPlane,
                                     SimpleFaultSurface)
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hmtk.faults.mfd.youngs_coppersmith import (
    YoungsCoppersmithCharacteristic, YoungsCoppersmithExponential)
from openquake.hmtk.faults.mfd.characteristic import Characteristic
from openquake.hmtk.strain.strain_utils import (moment_function,
                                                moment_magnitude_function)


def source_shapefile_to_dictionary(filename):
    """
    """
    sf = shapefile.Reader(filename)
    fields = [fld[0] for fld in sf.fields[1:]]
    data = []
    for rec in sf.shapeRecords():
        zone = dict([(key, val)
                     for key, val in zip(fields[1:], rec.record)])
        zone["geometry"] = Polygon([Point(pnt[0], pnt[1])
                                    for pnt in rec.shape.points])
        data.append(zone)
    return data

def fault_shapefile_to_dictionary(filename, mesh_spacing=1.0):
    """
    
    """
    sf = shapefile.Reader(filename)
    data = []
    fields = [val[0] for val in sf.fields[1:]]
    for rec in sf.shapeRecords():
        rec_data = dict([(field, rec.record[i])
                         for i, field in enumerate(fields)])
        # Parse geometry
        rec_data["geometry"] = np.array(rec.shape.points)
        data.append(rec_data)
    # Cleanup dict
    cleaned_data = []
    for rec in data:
        cleaned_rec = {}
        cleaned_rec["ID"] = rec["IDSOURCE"]
        cleaned_rec["dip"] = (rec["DIPMIN"], rec["DIPMAX"])
        cleaned_rec["name"] = rec["SOURCENAME"]
        cleaned_rec["rake"] = (rec["RAKEMIN"], rec["RAKEMAX"])
        cleaned_rec["LSD"] = rec["MAXDEPTH"]
        cleaned_rec["USD"] = rec["MINDEPTH"]
        cleaned_rec["slip"] = (rec["SRMIN"], rec["SRMAX"])
        # Build surfaces
        cleaned_rec["trace"] = Line([Point(row[0], row[1])
                                     for row in rec["geometry"]])
        cleaned_rec["surfaces"] = (
            SimpleFaultSurface.from_fault_data(cleaned_rec["trace"],
                                               cleaned_rec["USD"],
                                               cleaned_rec["LSD"],
                                               cleaned_rec["dip"][1], # Highest dip -> smallest area
                                               mesh_spacing),
            SimpleFaultSurface.from_fault_data(cleaned_rec["trace"],
                                               cleaned_rec["USD"],
                                               cleaned_rec["LSD"],
                                               cleaned_rec["dip"][0], # Lowest dip -> largest area
                                               mesh_spacing)
                                               )
        cleaned_data.append(cleaned_rec)
    return cleaned_data


def get_incremental_cumulative(mfd_dist):
    """
    Returns the incremental and cumulative recurrence rates (and the mfd node)
    """
    n_rates = len(mfd_dist.occurrence_rates)
    cum_rates = [sum(mfd_dist.occurrence_rates[i:]) for i in range(n_rates)]
    mags = mfd_dist.min_mag +\
        np.cumsum(mfd_dist.bin_width * np.ones(n_rates)) - mfd_dist.bin_width
    return mags, mfd_dist.occurrence_rates, cum_rates, mfd_dist


def build_fault_model(area, slip, mmax, config={}, mu=30.0, coupling=1.0):
    """
    
    """
    if not config:
        # Simple Dirac function
        geological_moment = coupling * (mu * 1.0E9) * (area * 1.0E6) * (slip * 1.0E-3)
        mmax_moment = moment_function(mmax)
        rate = geological_moment / mmax_moment
        mf_dist = EvenlyDiscretizedMFD(mmax, 0.1, [rate])
        return get_incremental_cumulative(mf_dist)

    d_m = config["MFD_spacing"]
    config["Model_Weight"] = 1.0

    if config["model"] == "Characteristic":
        # Try characteristic
        for key in ["MFD_spacing", "Lower_Bound", "Upper_Bound", "Sigma"]:
            # Verify everything is present
            assert key in config and config[key] is not None
        rec_model = Characteristic()
        rec_model.setUp(config)
        rec_model.mmax = mmax
    elif config["model"] == "Hybrid":
        # Try Hybrid
        for key in ["Minimum_Magnitude", "b_value"]:
            # Verify everything is present
            assert key in config and config[key] is not None
        rec_model = YoungsCoppersmithCharacteristic()
        rec_model.setUp(config)
        rec_model.mmax = mmax
    elif config["model"] == "Exponential":
        # Try exponential
        for key in ["Minimum_Magnitude", "b_value"]:
            # Verify everything is present
            assert key in config and config[key] is not None 
        rec_model = YoungsCoppersmithExponential()
        rec_model.setUp(config)
        rec_model.mmax = mmax
        rec_model.mmin += (config["MFD_spacing"] / 2.)
    else:
        raise ValueError("Model %s not recognised" % config["model"])

    mmin, bin_width, rate = rec_model.get_mfd(slip, area, mu)
    mf_dist = EvenlyDiscretizedMFD(mmin,
                                   bin_width, rate.tolist())
    return get_incremental_cumulative(mf_dist)
        

        


