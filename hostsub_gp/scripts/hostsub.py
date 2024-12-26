# hostsub_gp/scripts/hostsub.py
# The main script to conduct an end-to-end host subtraction

import numpy as np
from astropy.io import fits

import os

from pypeit import msgs
from pypeit.spectrographs.util import load_spectrograph
from pypeit.par import pypeitpar

from hostsub_gp import SpecData
from hostsub_gp._plt import plt
from .scriptbase import ScriptBase
from ..inputfiles import HostSubInput


class HostSub(ScriptBase):
    @classmethod
    def get_parser(cls):
        parser = super().get_parser(description="Run the host subtraction pipeline.")
        parser.add_argument(
            "hostsub_file", type=str, help="Input configuration file."
        )  # TODO: Add the syntax for the input configuration file
        parser.add_argument(
            "--overwrite",
            "-o",
            default=False,
            action="store_true",
            help="Re-do the rectification and overwrite the fits files.",
        )
        parser.add_argument(
            "--debug",
            default=False,
            action="store_true",
            help="Run the script in debug mode and present the QA plots.",
        )
        parser.add_argument(
            "--par_outfile",
            type=str,
            default="hostsub.par",
            help="Name of output file to save the parameters used by the GP.",
        )
        return parser

    @staticmethod
    def main(args):
        # Load the configuration file
        hostsubFile = HostSubInput.from_file(args.hostsub_file)
        par = hostsubFile.config

        # Prepare the QA directory
        os.system(f"mkdir -p QA")

        # Standard star
        if any(hostsubFile.data["frametype"] == "standard"):
            std_idx = np.argwhere(hostsubFile.data["frametype"] == "standard").ravel()
            std_file = hostsubFile.filenames[std_idx[0]]
        else:
            std_file = None

        # Loop over science files
        sci_idx = np.argwhere(hostsubFile.data["frametype"] == "science").ravel()
        for i in sci_idx:
            sci_file_1d = hostsubFile.filenames[i]
            sci_file_2d = sci_file_1d.replace("spec1d", "spec2d")
            sci_rect_file = sci_file_2d.replace(".fits", "_rect.fits").replace("spec1d", "spec2d")
            base_file = sci_file_1d.replace("spec1d_", "").replace(".fits", "")

            # If the object ID is not provided,
            # Set it to None and use the standard star file
            objid = hostsubFile.data["objid"][i]
            objid = None if len(objid) == 0 else objid

            if std_file is None and objid is None:
                msgs.error("No standard star file provided and no object ID given.")

            # Get the parameters for host subtraction
            par_hostsub = par.get("hostsub", {})
            spec2d_cfg = {}
            spec2d_cfg["slit_len"] = float(par_hostsub.get("slit_len", 20.0))
            spec2d_cfg["ra"] = None if not "ra" in par_hostsub else float(par_hostsub["ra"])
            spec2d_cfg["dec"] = None if not "dec" in par_hostsub else float(par_hostsub["dec"])

            # Run the host subtraction
            if args.overwrite or not os.path.exists(sci_rect_file):
                # Load the pypeit 2dspec file and save the rectified file
                spec_data = SpecData.from_pypeit(
                    sci_file=sci_file_2d,
                    std_file=std_file,
                    obj_id=objid,
                    **spec2d_cfg,
                )
            else:
                # Load the rectified file
                spec_data = SpecData.from_fits(sci_rect_file)

            # Convert the 2D spectrum to a SpecModel object
            hostsub_cfg = {}
            hostsub_cfg["spec_range"] = (
                None if "spec_range" not in par_hostsub else tuple(map(float, par_hostsub["spec_range"]))
            )
            hostsub_cfg["mask_wid"] = float(par_hostsub.get("mask_wid", 2.0))
            hostsub_cfg["sky_wid"] = float(par_hostsub.get("sky_wid", 10.0))
            hostsub_cfg["mask_offset"] = float(par_hostsub.get("mask_offset", 0.0))
            spec2d = spec_data.to_SpecModel(
                show=args.debug,
                save=f"QA/{os.path.basename(base_file)}.pdf",
                **hostsub_cfg,
            )

            # Model the host prior
            spec2d.model_host_prior(
                show=args.debug,
                save=f"QA/{os.path.basename(base_file)}_host_prior.pdf",
            )

            # Get the initial parameters
            params_init_1d = par_hostsub.get("params_init_1d", None)
            params_init_2d = par_hostsub.get("params_init_2d", None)
            params_init = [params_init_1d, params_init_2d]

            # Get limits for the parameters
            def _set_params_limit(params_limit_dict):
                """Integrate upper and lower limits of each parameter."""
                upper = {k.replace("_upper", ""): v for k, v in params_limit_dict.items() if "upper" in k}
                lower = {k.replace("_lower", ""): v for k, v in params_limit_dict.items() if "lower" in k}
                return {k: (lower[k], upper[k]) for k in lower}

            params_limit_1d = _set_params_limit(par_hostsub.get("params_limit_1d", None))
            params_limit_2d = _set_params_limit(par_hostsub.get("params_limit_2d", None))

            params_limit = [params_limit_1d, params_limit_2d]

            # Model the host
            spec2d.model_host(
                params_init=params_init,
                params_limit=params_limit,
                optimization=True,
                optimization_kwargs={"maxiter": 1000, "tol": 1e-2},
            )

            # QA plots
            # Raw, model, and residual
            spec2d._plot_pred()
            plt.savefig(f"QA/{os.path.basename(base_file)}_pred.pdf")
            if args.debug:
                plt.show()

            # Prior and posterior of the host profiles
            spec2d._plot_host_profile_pred()
            plt.savefig(f"QA/{os.path.basename(base_file)}_host_profile_pred.pdf")
            if args.debug:
                plt.show()

            # Extract the science spectrum
            spec2d.extract_sci()
            plt.savefig(f"QA/{os.path.basename(base_file)}_sci.pdf")
            if args.debug:
                plt.show()
