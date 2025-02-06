# hostsub_gp/inputfiles.py

from pypeit.inputfiles import InputFile

class HostSubInput(InputFile):
    data_block = "hostsub"
    flavor = "HostSub"
    setup_required = False
    datablock_required = True
    required_columns = ["filename", "objid", "frametype"]