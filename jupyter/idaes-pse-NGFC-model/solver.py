import os
import logging
import tarfile
import idaes
from shutil import copyfile
from pyomo.common.download import FileDownloader

_log = logging.getLogger(__name__)

def download_binaries(url=None, verbose=False):
    """
    Download IDAES solvers and libraries and put them in the right location. Need
    to supply either local or url argument.

    Args:
        url (str): a url to download binary files to install files

    Returns:
        None
    """
    if verbose:
        _log.setLevel(logging.DEBUG)
    idaes._create_lib_dir()
    idaes._create_bin_dir()
    solvers_tar = os.path.join(idaes.bin_directory, "idaes-solvers.tar.gz")
    libs_tar = os.path.join(idaes.lib_directory, "idaes-lib.tar.gz")
    fd = FileDownloader()
    arch = fd.get_sysinfo()
    if url is not None:
        if not url.endswith("/"):
            c = "/"
        else:
            c = ""
        solvers_from = c.join([url, "idaes-solvers-{}-{}.tar.gz".format(arch[0], arch[1])])
        libs_from = c.join([url, "idaes-lib-{}-{}.tar.gz".format(arch[0], arch[1])])
        _log.debug("URLs \n  {}\n  {}\n  {}".format(url, solvers_from, libs_from))
        _log.debug("Destinations \n  {}\n  {}".format(solvers_tar, libs_tar))
        if arch[0] == 'darwin':
            raise Exception('Mac OSX currently unsupported')
        fd.set_destination_filename(solvers_tar)
        fd.get_binary_file(solvers_from)
        fd.set_destination_filename(libs_tar)
        fd.get_binary_file(libs_from)
    else:
        raise Exception("Must provide a location to download binaries")

    _log.debug("Extracting files in {}".format(idaes.bin_directory))
    with tarfile.open(solvers_tar, 'r') as f:
        f.extractall(idaes.bin_directory)
    _log.debug("Extracting files in {}".format(idaes.lib_directory))
    with tarfile.open(libs_tar, 'r') as f:
        f.extractall(idaes.lib_directory)
