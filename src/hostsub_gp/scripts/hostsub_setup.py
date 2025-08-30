# hostsub_gp/scripts/hostsub_setup.py
# Setup script for HostSub_GP: reading from config files and data products of upstream pipelines

import os
import argparse
import glob
import numpy as np

import io
from astropy.io import ascii
from astropy.table import Table

from typing import Any

from .scriptbase import ScriptBase
from ..inputfiles import HostSubInput, Digitize
from .._utils import msgs


class HostSubSetup(ScriptBase):
    """Setup HostSub_GP run by reading configuration files and upstream pipeline data products.

    This script reads in a configuration file and data products from upstream pipelines (e.g., PypeIt),
    prepares the necessary inputs, and initializes the HostSub_GP processing environment.
    """

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Setup HostSub_GP with configuration files and upstream pipeline data products."
        )
        parser.add_argument(
            "upstream_pipeline",
            type=str,
            help="Name of the upstream pipeline (e.g., 'PypeIt')",
        )
        parser.add_argument(
            "--config_file",
            "-c",
            type=str,
            help="Path to the configuration file for HostSub_GP",
        )
        parser.add_argument(
            "--target",
            type=str,
            nargs="+",
            help="Names of the target objects",
        )
        return parser

    @staticmethod
    def main(args: argparse.Namespace):
        if args.upstream_pipeline.lower() == "pypeit":
            from pypeit import specobjs
            from pypeit.inputfiles import PypeItFile

            if args.config_file is None or args.target is None:
                raise ValueError(
                    "Both --config_file and --obj must be provided for PypeIt."
                )

            dict_out = {"filename": [], "frametype": [], "objid": []}

            # Load the data block from PypeIt files
            pypeit_file = PypeItFile.from_file(args.config_file)
            pypeit_data_block = pypeit_file.data

            # Default user-level parameters for host subtraction
            cfg_lines = [
                "[hostsub]",
                "    slit_len = 60.0",
                "    # ra = PLEASE_SET_RA",
                "    # dec = PLEASE_SET_DEC",
                "    mask_wid = 2.5",
                "    sky_region = -10, 10.",
                "    host_wid = 10.0",
                "    mask_offset = 0.0",
                "    sky_offset = 0.0",
                "    batch_2d = 2, 256",
                f"    raw_dir = {pypeit_file.file_paths[0]}",
                "    spat_resln = 1.0",
                "",
                "    [[host_prior]]",
                "        survey = PS1",
                "        filters = grizy",
                "",
                "    [[host_emission]]",
                "        find_host_emission = True",
                "        # z = 0.0",
                "",
                "    [[seeing_match]]",
                "        dseeing_lower = 0.0",
                "        dseeing_upper = 1.0",
            ]

            idx_std = pypeit_data_block[pypeit_data_block["frametype"] == "standard"]
            if len(idx_std) > 0:
                target_std = idx_std["target"][0]
                msgs.info(f"Standard star identified: {target_std}")
            else:
                target_std = None
                msgs.warning("No standard star found in the PypeIt file.")

            pypeit_files = sorted(glob.glob("./Science/spec1d*[0-9].fits"))

            for target in args.target:
                msgs.info(f"Setting up HostSub_GP for object: {target}")
                os.makedirs(f"HostSub_{target}", exist_ok=True)

                pypeit_sci_files = [f for f in pypeit_files if target in f]
                if target_std is None:
                    pypeit_std_files = []
                else:
                    pypeit_std_files = [f for f in pypeit_files if target_std in f]

                if len(pypeit_std_files) == 0:
                    spat_id_std, slit_id_std, det_id_std = -1, -1, -1

                for std_file in pypeit_std_files:
                    dict_out["filename"].append(std_file.split("/")[-1])
                    dict_out["frametype"].append("standard")

                    std_objs = specobjs.SpecObjs.from_fitsfile(std_file)
                    if len(std_objs) == 0:
                        msgs.warning(
                            f"No standard objects found in {std_file}. Skipping."
                        )
                        dict_out["objid"].append("")
                        continue

                    # Find the SpecObj with the highest signal-to-noise ratio (S2N) in the SpecObjs
                    objs2n = []
                    for obj in std_objs:
                        assert obj is not None
                        objs2n.append(obj["S2N"])
                    std_obj = std_objs[np.argmax(objs2n)]

                    dict_out["objid"].append(std_obj["NAME"])
                    msgs.info(
                        f"Using standard object with highest S2N: {std_obj['NAME']}"
                    )

                    # Extract spatial, slit, and detector IDs from the standard object's name
                    spat_id_std, slit_id_std, det_id_std = std_obj.NAME.split("-")
                    spat_id_std = int(spat_id_std[-4:])
                    slit_id_std = int(slit_id_std[-4:])
                    det_id_std = int(det_id_std[-2:])

                for sci_file in pypeit_sci_files:
                    dict_out["filename"].append(sci_file.split("/")[-1])
                    dict_out["frametype"].append("science")

                    traceobj = specobjs.SpecObjs.from_fitsfile(sci_file)
                    if len(traceobj) == 0:
                        msgs.warning(f"No trace objects found in {sci_file}. Skipping.")
                        dict_out["objid"].append("")
                        continue

                    # Find the SpecObj that matches the slit and detector IDs
                    # and the spatial position nearest to the standard star
                    sci_objs = []
                    sci_spat_ids = []
                    for obj in traceobj:
                        assert obj is not None
                        spat_id, slit_id, det_id = obj.NAME.split("-")
                        spat_id = int(spat_id[-4:])
                        slit_id = int(slit_id[-4:])
                        det_id = int(det_id[-2:])
                        if slit_id == slit_id_std and det_id == det_id_std:
                            sci_objs.append(obj)
                            sci_spat_ids.append(spat_id)

                    if len(sci_objs) == 0:
                        msgs.warning(
                            f"No matching science objects found in {sci_file}. Skipping."
                        )
                        dict_out["objid"].append("")
                        continue
                    sci_obj = sci_objs[
                        np.argmin(np.abs(np.array(sci_spat_ids) - spat_id_std))
                    ]

                    dict_out["objid"].append(sci_obj["NAME"])
                    msgs.info(f"Finding science object: {sci_obj['NAME']}")

                data = Table(dict_out)

                with open(f"HostSub_{target}/hostsub.txt", "w") as f:
                    # Write the configuration block
                    for line in cfg_lines:
                        f.write(line + "\n")
                    f.write("\n")

                    # Write the data block
                    f.write("# Data block for input spectra\n")
                    f.write("hostsub read\n")
                    f.write(" path ../Science/\n")
                    buf = io.StringIO()
                    ascii.write(data, buf, format="fixed_width", delimiter="|")
                    for line in buf.getvalue().splitlines():
                        # Remove leading/trailing pipes and whitespace
                        f.write(line.strip("|").rstrip() + "\n")

                    f.write("hostsub end\n")

        else:
            raise NotImplementedError(
                f"Upstream pipeline '{args.upstream_pipeline}' is not supported yet."
            )
