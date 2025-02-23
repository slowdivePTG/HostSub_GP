# hostsub_gp/inputfiles.py

from pypeit.inputfiles import InputFile
from typing import Callable


class HostSubInput(InputFile):
    data_block = "hostsub"
    flavor = "HostSub"
    setup_required = False
    datablock_required = True
    required_columns = ["filename", "objid", "frametype"]


class Digitize:
    """
    Convert the input value to the specified data type.

    If the input contains multiple tuples, convert them to a list of tuples.
    """

    def __init__(self, _call: Callable) -> None:
        self._call = _call

    def __call__(self, value) -> any:
        if isinstance(value, list):
            if isinstance(value[0], str):
                # The input contains brackets
                if "(" in value[0]:
                    if len(value) % 2 != 0:
                        raise ValueError("The length of the tuple list must be even.")
                    value = ",".join(value).replace("(", "").replace(")", "").split(",")
                    return [(self._call(value[i]), self._call(value[i + 1])) for i, _ in enumerate(value) if i % 2 == 0]
            # The input contains only values that can be directly converted to the specified data type.
            return [self._call(v) for v in value]

        elif (value == "None") or (value == "none") or value is None:
            return None
        else:
            return self._call(value)
