# hostsub_gp/scripts/scriptbase.py
# mostly adopted from PypeIt.pypeit.scripts.scriptbase

import os
import argparse


class ScriptBase:
    """Base class for scripts."""

    @classmethod
    def entry_point(cls):
        """Entry point for script."""
        cls.main(cls.parse_args())

    @classmethod
    def parse_args(cls, options=None):
        """
        Parse the command-line arguments.
        """
        parser = cls.get_parser()
        ScriptBase._fill_parser_cwd(parser)
        return parser.parse_args() if options is None else parser.parse_args(options)

    @staticmethod
    def _fill_parser_cwd(parser: argparse.ArgumentParser):
        """
        Replace the default of any action that is exactly ``'current working
        directory'`` with the value of ``os.getcwd()``.

        The ``parser`` is edited *in place*.

        Args:
            parser (argparse.ArgumentParser):
                The argument parsing object to edit.
        """
        for action in parser._actions:
            if action.default == "current working directory":
                action.default = os.getcwd()

    @classmethod
    def get_parser(cls, description: str = None) -> argparse.ArgumentParser:
        """
        Construct the command-line argument parser.

        Derived classes should override this.  Ideally they should use this
        base-class method to instantiate the ArgumentParser object and then fill
        in the relevant parser arguments

        .. warning::

            *Any* argument that defaults to the
            string ``'current working directory'`` will be replaced by the
            result of ``os.getcwd()`` when the script is executed.  This means
            help dialogs will include this replacement, and parsing of the
            command line will use ``os.getcwd()`` as the default.  This
            functionality is largely to allow for PypeIt's automated
            documentation of script help dialogs without the "current working"
            directory being that of the developer that most recently compiled
            the docs.
        """
        return argparse.ArgumentParser(description=description)

    @staticmethod
    def main(args: argparse.Namespace):
        """
        Execute the script.

        Derived classes should override this to perform the desired.
        """
        pass
