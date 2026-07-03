# hostsub_gp/scripts/hostsub.py
# The main script to conduct an end-to-end host subtraction

from typing import Callable
import numpy as np
import sys

import os
import argparse

from astropy.io import fits

from typing import Any

from hostsub_gp import SpecData, SpecModel
from .scriptbase import ScriptBase
from ..inputfiles import HostSubInput, Digitize
from .._utils import msgs


Float: Callable[[Any], float | tuple[float, float]] = Digitize(float)
Int: Callable[[Any], int | tuple[int, int]] = Digitize(int)


class HostSub(ScriptBase):
    @staticmethod
    def _hostsub_product_path(path: str, suffix: str = "") -> str:
        """
        Return the HostSub product path in a sibling Science_hostsub directory.
        """
        dirname, basename = os.path.split(path)
        parent, dirname_base = os.path.split(dirname)
        if dirname_base == "Science":
            out_dir = os.path.join(parent, "Science_hostsub")
        else:
            out_dir = os.path.join(dirname, "Science_hostsub")
        os.makedirs(out_dir, exist_ok=True)
        root, ext = os.path.splitext(basename)
        return os.path.join(out_dir, f"{root}{suffix}{ext}")

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
            "--skip_seeing_match",
            default=False,
            action="store_true",
            help="Skip the matching of the seeing between the host and science spectra.",
        )
        parser.add_argument(
            "--skip_gp",
            default=False,
            action="store_true",
            help="Skip the GP modeling (only preprocess the spectrum and perform classic host subtraction).",
        )
        parser.add_argument(
            "--coadd2d",
            default=False,
            action="store_true",
            help="Coadd the 2D spectra before modeling the host galaxy.",
        )
        parser.add_argument(
            "--write_spec2d",
            default=False,
            action="store_true",
            help="Also write HostSub spec2d products when using --coadd2d.",
        )
        return parser

    @msgs.timer
    @staticmethod
    def main(args: argparse.Namespace):
        # Load the configuration file
        hostsubFile = HostSubInput.from_file(args.hostsub_file)
        assert hostsubFile.data is not None, "No data found in the configuration file."
        assert hostsubFile.filenames is not None, (
            "No files found in the configuration file."
        )
        assert hostsubFile.config is not None
        par = hostsubFile.config

        # Prepare the QA directory
        os.makedirs("QA", exist_ok=True)

        # Standard star
        if any(hostsubFile.data["frametype"] == "standard"):
            std_idx = np.argwhere(hostsubFile.data["frametype"] == "standard").ravel()
            std_file = hostsubFile.filenames[std_idx[0]]
        else:
            std_file = None

        # Loop over science files
        sci_idx = np.argwhere(hostsubFile.data["frametype"] == "science").ravel()
        if len(sci_idx) == 0:
            msgs.error("No science files found in the configuration file.")
        spec_data_list = []
        spat_rect, spec_rect = (
            None,
            None,
        )  # For all the science files, use the same points for interpolation
        base_file_list = []
        for i in sci_idx:
            sci_file_1d = hostsubFile.filenames[i]
            sci_file_2d = sci_file_1d.replace("spec1d", "spec2d")
            sci_rect_file = HostSub._hostsub_product_path(sci_file_2d, "_rect")
            sci_preproc_file = HostSub._hostsub_product_path(sci_file_2d, "_preproc")
            base_file_list.append(
                sci_file_1d.replace("spec1d_", "").replace(".fits", "")
            )

            # If the object ID is not provided,
            # Set it to None and use the standard star file
            objid = hostsubFile.data["objid"][i]
            objid = None if len(objid) == 0 else objid

            if std_file is None and objid is None:
                msgs.error("No standard star file provided and no object ID given.")

            # Get the parameters for host subtraction
            par_hostsub = par.get("hostsub", {})
            par_host_prior = par_hostsub.get("host_prior", {})
            raw_dir = par_hostsub.get("raw_dir", None)
            spec_data_cfg = {}
            spec_data_cfg["ra"] = Float(par_hostsub.get("ra", None))
            spec_data_cfg["dec"] = Float(par_hostsub.get("dec", None))
            spec_data_cfg["sky_offset"] = Float(par_hostsub.get("sky_offset", None))
            spec_data_cfg["spat_resln"] = Float(par_hostsub.get("spat_resln", None))
            spec_data_cfg["survey"] = par_host_prior.get("survey", "PS1")

            # Run the host subtraction
            from_pypeit = args.overwrite or not os.path.exists(sci_rect_file)
            if from_pypeit:
                # Load the pypeit 2dspec file and save the rectified file
                spec_data = SpecData.from_pypeit(
                    sci_file=sci_file_2d,
                    raw_dir=raw_dir,
                    std_file=std_file,
                    obj_id=objid,
                    spat_rect=spat_rect,
                    spec_rect=spec_rect,
                    rect_file=sci_rect_file,
                    preproc_file=sci_preproc_file,
                    **spec_data_cfg,
                )
                spec_rect = spec_data.spec_rect
                spat_rect = spec_data.spat_rect
            else:
                # Load the rectified file
                spec_data = SpecData.from_fits(sci_rect_file)
            spec_data_list.append(spec_data)

        if args.coadd2d:
            spec_data_coadd2d, spec_data_cr_mask = SpecData.coadd2d(
                spec_data_list,  # show=from_pypeit
            )  # Only show the plot after a new rectification

            # Write the cr_mask to the rectified files
            for k, i in enumerate(sci_idx):
                sci_file_1d = hostsubFile.filenames[i]
                sci_file_2d = sci_file_1d.replace("spec1d", "spec2d")
                sci_rect_file = HostSub._hostsub_product_path(sci_file_2d, "_rect")
                hdul_rect = fits.open(sci_rect_file, mode="update")

                if spec_data_cr_mask is not None:
                    # Create a new frame for the CR mask
                    hdu_cr_mask = fits.ImageHDU(
                        data=np.array(spec_data_cr_mask[k], dtype=int), name="CR_MASK"
                    )

                    if "CR_MASK" in hdul_rect:
                        # Overwrite the existing CR mask
                        hdul_rect["CR_MASK"].data = np.array(
                            spec_data_cr_mask[k], dtype=int
                        )
                    else:
                        # Append the new CR mask to the HDU list
                        hdul_rect.append(hdu_cr_mask)

                    # Update the header with the new CR mask
                    hdul_rect[0].header["CR_MASK"] = True

                # Save the updated file
                hdul_rect.writeto(sci_rect_file, overwrite=True)

                hdul_rect.close()

            spec_model = HostSub._model_host_subtraction(
                args, spec_data_coadd2d, par_hostsub, output_suffix="coadd2d"
            )
            spec_model_for_rewrite = spec_model
            if not args.skip_gp:
                from pypeit import coadd2d

                spec2d_files = [
                    hostsubFile.filenames[i].replace("spec1d", "spec2d")
                    for i in sci_idx
                ]
                coadd_basename = coadd2d.CoAdd2D.default_basename(spec2d_files)
                coadd_spec1d_file = os.path.join(
                    os.path.dirname(
                        HostSub._hostsub_product_path(hostsubFile.filenames[sci_idx[0]])
                    ),
                    f"spec1d_{coadd_basename}_hostsub.fits",
                )
                objid = hostsubFile.data["objid"][sci_idx[0]]
                objid = None if len(objid) == 0 else objid
                pypeit_local_sky = {
                    "enabled": True,
                    "bsp": 0.6,
                    "sigrej": 3.5,
                    "npoly": 1,
                    "bkpts_optimal": True,
                }
                pypeit_local_sky.update(par_hostsub.get("pypeit_local_sky", {}))
                SpecData.write_pypeit_spec1d(
                    spec_model=spec_model,
                    template_spec1d=hostsubFile.filenames[sci_idx[0]],
                    output_file=coadd_spec1d_file,
                    obj_id=objid,
                    pypeit_local_sky=pypeit_local_sky,
                )

        else:
            for spec_data, base_file in zip(spec_data_list, base_file_list):
                spec_model = HostSub._model_host_subtraction(
                    args, spec_data, par_hostsub, output_suffix=base_file.split("/")[-1]
                )
            # For multi-file non-coadd runs, intentionally use the final model for
            # all rewrites; the updated 2D products are expected to be coadded.
            spec_model_for_rewrite = spec_model
        if not args.skip_gp and ((not args.coadd2d) or args.write_spec2d):
            # Update the skymodel frame in the original Spec2D object
            for i, spec_data in zip(sci_idx, spec_data_list):
                sci_file_1d = hostsubFile.filenames[i]
                sci_file_2d = sci_file_1d.replace("spec1d", "spec2d")
                spec_data.update_pypeit_skymodel(
                    spec_model=spec_model_for_rewrite,
                    spec2d_file=sci_file_2d,
                    preproc_file=HostSub._hostsub_product_path(
                        sci_file_2d, "_preproc"
                    ),
                    rect_file=HostSub._hostsub_product_path(sci_file_2d, "_rect"),
                    output_file=HostSub._hostsub_product_path(
                        sci_file_2d, "_hostsub"
                    ),
                )

    @staticmethod
    def _model_host_subtraction(
        args: argparse.Namespace,
        spec_data: SpecData,
        par_hostsub: dict,
        output_suffix: str = None,
    ) -> SpecModel:
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

        Returns
        -------
        SpecModel
            The SpecModel object with the host galaxy subtracted.
        """

        # Convert the 2D spectrum to a SpecModel object
        # Parameters for defining the SpecModel object
        spec_model_cfg = {}
        spec_model_cfg["slit_len"] = Float(par_hostsub.get("slit_len", None))
        spec_model_cfg["slit_trim"] = Int(par_hostsub.get("slit_trim", [5, 5]))
        spec_model_cfg["spec_range"] = Float(par_hostsub.get("spec_range", None))
        spec_model_cfg["mask_wid"] = Float(par_hostsub.get("mask_wid", 2.0))
        # If only the host_wid is provided, the host region is centered at the transient trace
        if ("host_wid" in par_hostsub) and ("host_region" not in par_hostsub):
            host_wid = Float(par_hostsub.get("host_wid"))
            spec_model_cfg["host_region"] = Float([-host_wid / 2, host_wid / 2])
        # If the host_region is provided, use it directly
        else:
            spec_model_cfg["host_region"] = Float(
                par_hostsub.get("host_region", [-5.0, 5.0])
            )
        spec_model_cfg["sky_region"] = Float(par_hostsub.get("sky_region", [-5.0, 5.0]))
        spec_model_cfg["mask_offset"] = Float(par_hostsub.get("mask_offset", 0.0))
        spec_model_cfg["spat_resln"] = Float(par_hostsub.get("spat_resln", None))
        spec_model_cfg["spec_resln"] = Float(par_hostsub.get("spec_resln", None))

        par_host_emission = par_hostsub.get("host_emission", {})
        par_seeing_match = par_hostsub.get("seeing_match", {})
        par_host_prior = par_hostsub.get("host_prior", {})

        # Parameters for all the SpecWrapper attributes of the SpecModel object
        spec_wrapper_cfg = {}
        spec_wrapper_cfg["batch_2d"] = Int(par_hostsub.get("batch_2d", [2, 128]))
        spec_wrapper_cfg["sigma_clip"] = Float(par_hostsub.get("sigma_clip", 5.0))

        # Parameters for identifying host emission lines
        host_emission_cfg = {}
        host_emission_cfg["find_host_emission"] = (
            par_host_emission.get("find_host_emission", "True").lower() == "true"
        )
        host_emission_cfg["p_value"] = Float(par_host_emission.get("p_value", 0.05))
        host_emission_cfg["kernel_wid"] = (
            None
            if "kernel_wid" not in par_host_emission
            else Float(par_host_emission["kernel_wid"])
        )
        host_emission_cfg["z"] = (
            None if "z" not in par_host_emission else Float(par_host_emission["z"])
        )
        host_emission_cfg["z_err"] = (
            None
            if "z_err" not in par_host_emission
            else Float(par_host_emission["z_err"])
        )

        # Parameters for matching the seeing of the host and science spectra
        seeing_match_cfg = {}
        seeing_match_cfg["dseeing_upper"] = Float(
            par_seeing_match.get("dseeing_upper", 1.0)
        )
        seeing_match_cfg["dseeing_lower"] = Float(
            par_seeing_match.get("dseeing_lower", 0.0)
        )
        seeing_match_cfg["dseeing"] = Float(par_seeing_match.get("dseeing", None))
        if seeing_match_cfg["dseeing"] == 0:
            seeing_match_cfg["dseeing"] = None

        # Parameters for modeling the host prior
        host_prior_cfg = {}
        host_prior_cfg["filters"] = par_host_prior.get("filters", "grizy")
        host_prior_cfg["survey"] = par_host_prior.get("survey", "PS1")
        # host_prior_cfg["spat_resln"] = Float(par_host_prior.get("spat_resln", 1.0))
        # host_prior_cfg["noise_smooth_kernel"] = Int(par_host_prior.get("noise_smooth_kernel", None))

        # Initialize the SpecModel object
        spec_model = spec_data.to_SpecModel(**spec_model_cfg)

        # Model the host prior
        spec_model.build_host_prior(
            from_archival=True,
            **host_prior_cfg,
            save=f"QA/{output_suffix}_host_prior.pdf",
        )

        # Initialize the raw spectra
        spec_model.construct_spec_wrapper(
            f_obs=spec_model.f_obs,
            host_emission_cfg=host_emission_cfg,
            **spec_wrapper_cfg,
            save=f"QA/{output_suffix}_raw.pdf",
        )

        # Get the classic extraction of the science spectrum
        spec_model.extract_sci_classic()

        # Skip the subsequent modeling if requested
        if args.skip_gp:
            # Extract the science spectrum
            spec_model.extract_sci(
                show=False, save=f"QA/{output_suffix}_classic_sci.pdf"
            )
            np.savetxt(
                f"QA/{output_suffix}_classic_sci.txt",
                np.array(
                    [
                        spec_model.f_sci_linear_1d.X.ravel(),
                        spec_model.f_sci_linear_1d.y,
                        spec_model.f_sci_linear_1d.yerr,
                        spec_model.f_sci_bspline_1d.y,
                        spec_model.f_sci_bspline_1d.yerr,
                    ]
                ).T,
                fmt="%.4f %.6e %.6e %.6e %.6e",
            )
            msgs.info(
                f"Saving the extracted science spectrum to QA/{output_suffix}_sci.txt"
            )
            return

        if not args.skip_seeing_match:
            dseeing = seeing_match_cfg.pop("dseeing", None)
            if dseeing is not None:
                msgs.info(
                    f"Using the seeing difference of {dseeing} provided by the user."
                )

            # Update the SpecModel object
            dseeing_opt, alpha_opt = spec_model.update_seeing(
                dseeing, **seeing_match_cfg
            )

            if dseeing_opt > 0:
                msgs.info(
                    "The seeing of the spectrum is better than the reference image."
                )
                msgs.info("Convolve the spectrum with a Gaussian kernel.")
                msgs.info(f"Optimized seeing difference is {dseeing_opt:.2f}.")

                # Update the SpecWrapper objects
                dseeing_wv = (
                    dseeing_opt
                    / spec_model.pixel_scale
                    * (spec_model.spec / spec_model.spec.mean()) ** (-alpha_opt)
                )
                spec_model.construct_spec_wrapper(
                    f_obs=spec_model.f_obs.fill_nan().convolve(dseeing_wv),
                    host_emission_cfg=host_emission_cfg,
                    **spec_wrapper_cfg,
                    save=f"QA/{output_suffix}_conv.pdf",
                )

            elif dseeing_opt < 0:
                msgs.info(
                    "The seeing of the spectrum is worse than the reference image."
                )
                msgs.info("Convolve the reference image with a Gaussian kernel.")
                msgs.info(f"Optimized seeing difference is {dseeing_opt:.2f}.")

                # Update the Host galaxy prior
                spec_model.build_host_prior(
                    from_archival=True,
                    **host_prior_cfg,
                    dseeing=dseeing_opt,
                    save=f"QA/{output_suffix}_host_prior_conv.pdf",
                )

            else:
                msgs.info(
                    "The seeing of the spectrum is the same as the reference image."
                )
                msgs.info("No convolution needed.")

        # Prior of the host profiles
        spec_model._plot_host_profile_prior(
            show=False, save=f"QA/{output_suffix}_host_profile_prior.pdf"
        )

        params_init, params_limit = HostSub._load_gp_params(par_hostsub)

        # Model the host
        spec_model.model_host(
            params_init=params_init,
            params_limit=params_limit,
            optimization=True,
            # optimization_kwargs={"maxiter": 1000, "tol": 1e-4},
        )

        # QA plots
        # Raw, model, and residual
        spec_model._plot_pred(show=False, save=f"QA/{output_suffix}_pred.pdf")

        # Posterior of the host profiles
        spec_model._plot_host_profile_pred(
            show=False, save=f"QA/{output_suffix}_host_profile_pred.pdf"
        )

        # Extract the science spectrum
        spec_model.extract_sci(show=False, save=f"QA/{output_suffix}_sci.pdf")
        np.savetxt(
            f"QA/{output_suffix}_sci.txt",
            np.array(
                [
                    spec_model.f_sci_pred_1d.X.ravel(),
                    spec_model.f_sci_pred_1d.y,
                    spec_model.f_sci_pred_1d.yerr,
                    spec_model.f_sci_bspline_1d.y,
                    spec_model.f_sci_bspline_1d.yerr,
                ]
            ).T,
            fmt="%.4f %.6e %.6e %.6e %.6e",
        )
        msgs.info(
            f"Saving the extracted science spectrum to QA/{output_suffix}_sci.txt"
        )

        return spec_model

    @staticmethod
    def _load_gp_params(
        par: dict,
    ) -> tuple[dict[str, float], tuple[dict[str, tuple[float, float]]]]:
        # Convert the initial parameters to the correct data type
        params_init_1d = {k: Float(v) for k, v in par.get("params_init_1d", {}).items()}
        params_init_2d = {k: Float(v) for k, v in par.get("params_init_2d", {}).items()}
        params_init = (params_init_1d, params_init_2d)

        # Get limits for the parameters
        # Reset the key names
        def _set_params_limit(params_limit_dict):
            """Integrate upper and lower limits of each parameter."""
            upper = {
                k.replace("_upper", ""): Float(v)
                for k, v in params_limit_dict.items()
                if "upper" in k
            }
            lower = {
                k.replace("_lower", ""): Float(v)
                for k, v in params_limit_dict.items()
                if "lower" in k
            }
            return {k: (lower[k], upper[k]) for k in lower}

        params_limit_1d = _set_params_limit(par.get("params_limit_1d", {}))
        params_limit_2d = _set_params_limit(par.get("params_limit_2d", {}))

        params_limit = (params_limit_1d, params_limit_2d)

        return params_init, params_limit
