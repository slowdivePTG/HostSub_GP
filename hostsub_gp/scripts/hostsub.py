# hostsub_gp/scripts/hostsub.py
# The main script to conduct an end-to-end host subtraction

import numpy as np

import os
import argparse

from hostsub_gp import SpecData
from .scriptbase import ScriptBase
from ..inputfiles import HostSubInput
from .._utils import plt, msgs


def Float(value: int | float | str) -> float:
    if (value == "None") or (value == "none") or value is None:
        return None
    else:
        return float(value)


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
            "--par_outfile",
            type=str,
            default="hostsub.par",
            help="Name of output file to save the parameters used by the GP.",
        )
        parser.add_argument(
            "--skip_model",
            default=False,
            action="store_true",
            help="Skip the modeling of the host galaxy (only load and rectify the spectrum).",
        )
        parser.add_argument(
            "--coadd2d",
            default=False,
            action="store_true",
            help="Coadd the 2D spectra before modeling the host galaxy.",
        )
        return parser

    @msgs.timer
    @staticmethod
    def main(args: argparse.Namespace):
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
        spec_data_list = []
        spec_rect = None  # For all the science files, use the same points for interpolation
        base_file_list = []
        for i in sci_idx:
            sci_file_1d = hostsubFile.filenames[i]
            sci_file_2d = sci_file_1d.replace("spec1d", "spec2d")
            sci_rect_file = sci_file_2d.replace(".fits", "_rect.fits").replace("spec1d", "spec2d")
            base_file_list.append(sci_file_1d.replace("spec1d_", "").replace(".fits", ""))

            # If the object ID is not provided,
            # Set it to None and use the standard star file
            objid = hostsubFile.data["objid"][i]
            objid = None if len(objid) == 0 else objid

            if std_file is None and objid is None:
                msgs.error("No standard star file provided and no object ID given.")

            # Get the parameters for host subtraction
            par_hostsub = par.get("hostsub", {})
            raw_dir = par_hostsub.get("raw_dir", None)
            spec2d_cfg = {}
            spec2d_cfg["slit_len"] = Float(par_hostsub.get("slit_len", 20.0))
            spec2d_cfg["ra"] = Float(par_hostsub.get("ra", None))
            spec2d_cfg["dec"] = Float(par_hostsub.get("dec", None))
            spec2d_cfg["sky_offset"] = Float(par_hostsub.get("sky_offset", None))

            # Run the host subtraction
            if args.overwrite or not os.path.exists(sci_rect_file):
                # Load the pypeit 2dspec file and save the rectified file
                spec_data = SpecData.from_pypeit(
                    sci_file=sci_file_2d,
                    raw_dir=raw_dir,
                    std_file=std_file,
                    obj_id=objid,
                    spec_rect=spec_rect,
                    **spec2d_cfg,
                )
                spec_rect = spec_data.spec_rect
            else:
                # Load the rectified file
                spec_data = SpecData.from_fits(sci_rect_file)
            spec_data_list.append(spec_data)

        if args.coadd2d:
            spec_data_coadd2d = SpecData.coadd2d(spec_data_list)
            HostSub._model_host_subtraction(args, spec_data_coadd2d, par_hostsub, output_suffix="coadd2d")
        else:
            for spec_data, base_file in zip(spec_data_list, base_file_list):
                HostSub._model_host_subtraction(args, spec_data, par_hostsub, output_suffix=base_file.split("/")[-1])

    @staticmethod
    def _model_host_subtraction(
        args: argparse.Namespace, spec_data: SpecData, par_hostsub: dict, output_suffix: str = None
    ):
        """
        Model the host galaxy and subtract it from the 1D spectrum.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments parsed by argparse.
        spec_data : SpecData
            2D spectrum data.
        par_hostsub : dict
            Parameters for host subtraction.
        """

        # Convert the 2D spectrum to a SpecModel object
        # Parameters for defining the SpecModel object
        host_sub_cfg = {}
        host_sub_cfg["slit_len"] = Float(par_hostsub.get("slit_len", 20.0))
        host_sub_cfg["spec_range"] = (
            None if "spec_range" not in par_hostsub else tuple(map(Float, par_hostsub["spec_range"]))
        )
        host_sub_cfg["host_wid"] = Float(par_hostsub.get("host_wid", 10.0))
        host_sub_cfg["mask_wid"] = Float(par_hostsub.get("mask_wid", 2.0))
        host_sub_cfg["sky_region"] = tuple(map(Float, par_hostsub.get("sky_region", (-5.0, 5.0))))
        host_sub_cfg["mask_offset"] = Float(par_hostsub.get("mask_offset", 0.0))
        host_sub_cfg["batch_2d"] = (
            (2, 128) if "batch_2d" not in par_hostsub else tuple(map(int, par_hostsub["batch_2d"]))
        )

        # Parameters for identifying host emission lines
        par_host_emission = par_hostsub.get("host_emission", {})
        host_emission_cfg = {}
        host_emission_cfg["find_host_emission"] = par_host_emission.get("find_host_emission", "True") in [
            "True",
            "true",
        ]
        host_emission_cfg["p_value"] = Float(par_host_emission.get("p_value", 0.05))
        host_emission_cfg["kernel_wid"] = (
            None if "kernel_wid" not in par_host_emission else Float(par_host_emission["kernel_wid"])
        )
        host_emission_cfg["z"] = None if "z" not in par_host_emission else Float(par_host_emission["z"])
        host_emission_cfg["z_err"] = None if "z_err" not in par_host_emission else Float(par_host_emission["z_err"])

        spec_model = spec_data.to_SpecModel(
            save=f"QA/{output_suffix}_raw.pdf",
            host_emission_cfg=host_emission_cfg,
            **host_sub_cfg,
        )

        # Model the host prior
        spec_model.model_host_prior(
            filters=par_hostsub.get("filters", "ugrizy"),
            save=f"QA/{output_suffix}_host_prior.pdf",
        )

        # Skip the subsequent modeling if requested
        if args.skip_model:
            return

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

        params_limit_1d = _set_params_limit(par_hostsub.get("params_limit_1d", {}))
        params_limit_2d = _set_params_limit(par_hostsub.get("params_limit_2d", {}))

        params_limit_1d["log_scale"] = params_limit_1d.get(
            "log_scale",
            np.log10(
                [
                    # lower bound
                    [1e1, spec_model.spec_resln / 2.355 / 2],
                    # upper bound
                    [1e3, spec_model.spec_resln * 2],
                ]
            ),
        )
        params_limit_2d["log_scale"] = params_limit_2d.get(
            "log_scale",
            np.log10(
                [
                    # lower bound
                    [spec_model.spat_resln / 2.355, spec_model.spec_resln / 2.355],
                    # upper bound
                    [spec_model.spat_resln, 1e4],
                ]
            ),
        )

        params_limit = [params_limit_1d, params_limit_2d]

        # Model the host
        spec_model.model_host(
            params_init=params_init,
            params_limit=params_limit,
            optimization=True,
            optimization_kwargs={"maxiter": 1000, "tol": 1e-2},
        )

        # QA plots
        # Raw, model, and residual
        spec_model._plot_pred()
        plt.savefig(f"QA/{output_suffix}_pred.pdf")
        plt.close()
        msgs.info(f"Saving the raw, model, and residual plots to QA/{output_suffix}_pred.pdf")

        # Prior and posterior of the host profiles
        spec_model._plot_host_profile_prior()
        plt.savefig(f"QA/{output_suffix}_host_profile_prior.pdf")
        plt.close()
        msgs.info(f"Saving the prior of the host profiles to QA/{output_suffix}_host_profile_prior.pdf")

        spec_model._plot_host_profile_pred()
        plt.savefig(f"QA/{output_suffix}_host_profile_pred.pdf")
        plt.close()
        msgs.info(f"Saving the posterior of the host profiles to QA/{output_suffix}_host_profile_pred.pdf")

        # Extract the science spectrum
        spec_model.extract_sci()
        np.savetxt(
            f"QA/{output_suffix}_sci.txt",
            np.array([spec_model.f_sci_pred_1d.X.ravel(), spec_model.f_sci_pred_1d.y, spec_model.f_sci_pred_1d.yerr]).T,
            fmt="%.4f %.6e %.6e",
        )
        plt.savefig(f"QA/{output_suffix}_sci.pdf")
        plt.close()
        msgs.info(f"Saving the extracted science spectrum to QA/{output_suffix}_sci.txt")
        msgs.info(f"Saving the extracted science spectrum plot to QA/{output_suffix}_sci.pdf")
