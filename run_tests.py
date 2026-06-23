#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np

import batch
from clawpack.geoclaw.surge.storm import Storm

scratch_dir = os.path.join(os.environ["CLAW"], 'geoclaw', 'scratch')

str_val = lambda value: str(int(value * 10)).zfill(2)

class ETCJob(batch.Job):

    def __init__(self, storm_path, article=False, scaling=1.0, 
                       sea_level=0.0, levels=2):

        super(ETCJob, self).__init__()

        self.scaling = scaling

        self.type = "surge"
        self.name = "ETC_storms"
        self.prefix = f"WS{str_val(scaling)}_SL{str_val(sea_level)}_L{levels}"
        self.prefix += f"_{storm_path.stem}"
        self.executable = "xgeoclaw"

        # Create base data object
        import setrun
        self.rundata = setrun.setrun()
        self.rundata.amrdata.amr_levels_max = levels
        self.rundata.geo_data.sea_level = sea_level

        # Path to storm data
        self.storm_path = storm_path

    def __str__(self):
        output = super(ETCJob, self).__str__()
        output += f"  storm path: {self.storm_path}\n"
        output += f"  Scaling: {self.scaling}\n"
        output += f"  Sea-Level: {self.rundata.geo_data.sea_level}\n"
        output += f"  Max Levels: {self.rundata.amrdata.amr_levels_max}\n"
        return output


    def write_data_objects(self, path):
        r""""""

        # Storm file lives in the job's run directory as "etc_storm.storm",
        # matching the path set in setrun.py.
        self.rundata.surge_data.storm_file = (path / "etc_storm.storm").resolve()

        # Write out storm.  Longitude wrapping and coordinate discovery are
        # handled by MetInspector at write time, so the NetCDF file can be
        # passed directly without any pre-wrapping step.
        etc_storm = Storm()
        etc_storm.file_paths.append(self.storm_path)
        etc_storm.time_offset = np.datetime64("2011-11-01T06:00:00.00")
        etc_storm.file_format = 'netcdf'
        etc_storm.scaling = [self.scaling, 1.0]   # [wind, pressure] scaling
        etc_storm.ramp_width = 2
        # Optional: restrict the forcing to a sub-window of the file.
        #   etc_storm.crop_extent = [lon0, lon1, lat0, lat1]
        # Coordinates are auto-discovered; only the non-standard wind/pressure
        # variable names need an explicit var_mapping.
        etc_storm.write(self.rundata.surge_data.storm_file,
                                        file_format='data',
                                        var_mapping={"wind_u": "U", "wind_v": "V",
                                                     "pressure": "P"},
                                        verbose=True)

        # Write out all data files
        super(ETCJob, self).write_data_objects(path)

    def post_run(self, result):
        r"""Plot each job once it finishes."""
        batch.plot_job(result, setplot="setplot.py", format="binary")

if __name__ == '__main__':

    # Currently a single storm; list structure left in place so additional
    # storms can be appended as the study expands (cf. ETC-NSLT project).
    base_path = (Path(os.environ['DATA_PATH']) / "storms" / "alaska").resolve()
    storm_paths = [base_path / "h01_output" / "uvp_latlon.nc"]

    jobs = []
    for sea_level in [0.0]:
        # for amr_max_levels in [2, 7]:
            # for scaling in [1.0, 1.1, 1.2]:
        for amr_max_levels in [7]:
            # for scaling in [1.2]:
            for scaling in [1.0]:
                for storm_path in storm_paths:
                    jobs.append(ETCJob(storm_path, 
                                       scaling=scaling, 
                                       sea_level=sea_level,
                                       levels=amr_max_levels))

    # ParallelExecutor is the default; per-job plotting is handled by
    # ETCJob.post_run.  run(wait=True) blocks until all jobs finish.
    controller = batch.BatchController(jobs)
    print(controller)
    controller.run(wait=True)
