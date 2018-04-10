# *- encoding: utf-8 -*-
"""
Utilities to download NeuroImaging datasets
"""

import os
import urllib
import urllib.request as urllib2
import tarfile
import zipfile
import sys
import shutil
import time
import hashlib
import fnmatch
import warnings

import numpy as np
from scipy import ndimage
from sklearn.datasets.base import Bunch

import nibabel

def _format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


class ResumeURLOpener(urllib2.FancyURLopener):
    """Create sub-class in order to overide error 206.  This error means a
       partial file is being sent, which is fine in this case.
       Do nothing with this error.

       Note
       ----
       This was adapted from:
       http://code.activestate.com/recipes/83208-resuming-download-of-a-file/
    """
    def http_error_206(self, url, fp, errcode, errmsg, headers, data=None):
        pass


def _chunk_report_(bytes_so_far, total_size, t0):
    """Show downloading percentage.

    Parameters
    ----------
    bytes_so_far: int
        Number of downloaded bytes

    total_size: int, optional
        Total size of the file. None is valid

    t0: int, optional
        The time in seconds (as returned by time.time()) at which the
        download was started.
    """
    if total_size:
        percent = float(bytes_so_far) / total_size
        percent = round(percent * 100, 2)
        dt = time.time() - t0
        # We use a max to avoid a division by zero
        remaining = (100. - percent) / max(0.01, percent) * dt
        # Trailing whitespace is too erase extra char when message length
        # varies
        sys.stderr.write(
            "Downloaded %d of %d bytes (%0.2f%%, %s remaining)  \r"
            % (bytes_so_far, total_size, percent,
               _format_time(remaining)))
    else:
        sys.stderr.write("Downloaded %d of ? bytes\r" % (bytes_so_far))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=0):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: urllib.addinfourl
        Response to the download request in order to get file size

    local_file: file
        Hard disk file where data should be written

    chunk_size: int, optional
        Size of downloaded chunks. Default: 8192

    report_hook: bool
        Whether or not to show downloading advancement. Default: None

    initial_size: int, optional
        If resuming, indicate the initial size of the file

    Returns
    -------
    data: string
        The downloaded file.

    """
    if total_size is None:
        total_size = response.info().get_all('Content-Length')
    try:
        total_size = int(total_size) + initial_size
    except Exception as e:
        if verbose > 0:
            print("Warning: total size could not be determined.")
            if verbose > 1:
                print("Full stack trace: %s" % e)
        total_size = None
    bytes_so_far = initial_size

    t0 = time.time()
    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                sys.stderr.write('\n')
            break

        local_file.write(chunk)
        if report_hook:
            _chunk_report_(bytes_so_far, total_size, t0)

    return


def _get_dataset_dir(dataset_name, data_dir=None, folder=None,
                     create_dir=True):
    """ Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    create_dir: bool, optional
        If the directory does not exist, determine whether or not it is created

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    This function retrieve the datasets directory (or data directory) using
    the following priority :
    1. the keyword argument data_dir
    2. the environment variable NILEARN_DATA
    3. "nilearn_data" directory into the current working directory
    """
    if not data_dir:
        data_dir = os.getenv("NILEARN_DATA", os.path.join(os.getcwd(),
                             'nilearn_data'))
    data_dir = os.path.join(data_dir, dataset_name)
    if folder is not None:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def _uncompress_file(file_, delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: bool, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    print('extracting data from %s...' % file_)
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        processed = False
        if ext == '.zip':
            z = zipfile.ZipFile(file_)
            z.extractall(data_dir)
            z.close()
            processed = True
        elif ext == '.gz':
            import gzip
            gz = gzip.open(file_)
            out = open(filename, 'wb')
            shutil.copyfileobj(gz, out, 8192)
            gz.close()
            out.close()
            # If file is .tar.gz, this will be handle in the next case
            if delete_archive:
                os.remove(file_)
            file_ = filename
            filename, ext = os.path.splitext(file_)
            processed = True
        if ext in ['.tar', '.tgz', '.bz2']:
            tar = tarfile.open(file_, "r")
            tar.extractall(path=data_dir)
            tar.close()
            processed = True
        if not processed:
            raise IOError("Uncompress: unknown file extension: %s" % ext)
        if delete_archive:
            os.remove(file_)
        print('   ...done.')
    except Exception as e:
        print('Error uncompressing file: %s' % e)
        raise


def _fetch_file(url, data_dir, resume=True, overwrite=False,
                verbose=0):
    """Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    url: string
        Contains the url of the file to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool, optional
        If true, try to resume partially downloaded files

    overwrite: bool, optional
        If true and file already exists, delete it.

    verbose: int, optional
        Defines the level of verbosity of the output

    Returns
    -------
    files: string
        Absolute path of downloaded file.

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded files are
    removed.
    """
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = os.path.basename(url)
    # Eliminate vars if needed
    file_name = file_name.split('?')[0]
    temp_file_name = file_name + ".part"
    full_name = os.path.join(data_dir, file_name)
    temp_full_name = os.path.join(data_dir, temp_file_name)
    if os.path.exists(full_name):
        if overwrite:
            os.remove(full_name)
        else:
            return full_name
    if os.path.exists(temp_full_name):
        if overwrite:
            os.remove(temp_full_name)
    t0 = time.time()
    local_file = None
    initial_size = 0
    try:
        # Download data
        print('Downloading data from %s ...' % url)
        if resume and os.path.exists(temp_full_name):
            url_opener = ResumeURLOpener()
            # Download has been interrupted, we try to resume it.
            local_file_size = os.path.getsize(temp_full_name)
            # If the file exists, then only download the remainder
            url_opener.addheader("Range", "bytes=%s-" % (local_file_size))
            try:
                data = url_opener.open(url)
            except urllib2.HTTPError:
                # There is a problem that may be due to resuming. Switch back
                # to complete download method
                return _fetch_file(url, data_dir, resume=False,
                                   overwrite=False)
            local_file = open(temp_full_name, "ab")
            initial_size = local_file_size
        else:
            data = urllib2.urlopen(url)
            local_file = open(temp_full_name, "wb")
        _chunk_read_(data, local_file, report_hook=True,
                     initial_size=initial_size, verbose=verbose)
        # temp file must be closed prior to the move
        if not local_file.closed:
            local_file.close()
        shutil.move(temp_full_name, full_name)
        dt = time.time() - t0
        print('...done. (%i seconds, %i min)' % (dt, dt / 60))
    except urllib2.HTTPError as e:
        print('Error while fetching file %s.' \
            ' Dataset fetching aborted.' % file_name)
        if verbose > 0:
            print("HTTP Error:", e, url)
        raise
    except urllib2.URLError as e:
        print('Error while fetching file %s.' \
            ' Dataset fetching aborted.' % file_name)
        if verbose > 0:
            print("URL Error:", e, url)
        raise
    finally:
        if local_file is not None:
            if not local_file.closed:
                local_file.close()
    return full_name


def _fetch_files(dataset_name, files, data_dir=None, resume=True, folder=None,
                 verbose=0):
    """Load requested dataset, downloading it if needed or requested.

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    files: list of (string, string, dict)
        List of files and their corresponding url. The dictionary contains
        options regarding the files. Options supported are 'uncompress' to
        indicates that the file is an archive and 'move' if renaming the file
        or moving it to a subfolder is needed.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: bool, optional
        If true, try resuming download if possible

    folder: string, optional
        Folder in which the file must be fetched inside the dataset folder.

    Returns
    -------
    files: list of string
        Absolute paths of downloaded files on disk
    """
    # Determine data path
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, folder=folder)

    files_ = []
    for file_, url, opts in files:
        # Download the file if it exists
        abs_file = os.path.join(data_dir, file_)
        if not os.path.exists(abs_file):

            dl_file = _fetch_file(url, data_dir, resume=resume,
                                  verbose=verbose)
            if 'move' in opts:
                shutil.move(os.path.join(data_dir, dl_file),
                            os.path.join(data_dir, opts['move']))
                dl_file = os.path.join(data_dir, opts['move'])
            if 'uncompress' in opts:
                _uncompress_file(dl_file)
        if not os.path.exists(abs_file):
            raise IOError('An error occured while fetching %s' % file_)
        files_.append(abs_file)
    return files_


def fetch_miyawaki2008(data_dir=None, url=None, resume=True, verbose=0):
    """Download and loads Miyawaki et al. 2008 dataset (153MB)

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'func': string list
            Paths to nifti file with bold data
        'label': string list
            Paths to text file containing session and target data
        'mask': string
            Path to nifti general mask file

    References
    ----------
    `Visual image reconstruction from human brain activity
    using a combination of multiscale local image decoders
    <http://www.cell.com/neuron/abstract/S0896-6273%2808%2900958-6>`_,
    Miyawaki, Y., Uchida, H., Yamashita, O., Sato, M. A.,
    Morito, Y., Tanabe, H. C., ... & Kamitani, Y. (2008).
    Neuron, 60(5), 915-929.

    Notes
    -----
    This dataset is available on the `brainliner website
    <http://brainliner.jp/data/brainliner-admin/Reconstruct>`_

    See `additional information
    <http://www.cns.atr.jp/dni/en/downloads/
    fmri-data-set-for-visual-image-reconstruction/>`_
    """

    url = 'https://www.nitrc.org/frs/download.php' \
          '/5899/miyawaki2008.tgz?i_agree=1&download_now=1'
    opts = {'uncompress': True}

    # Dataset files

    # Functional MRI:
    #   * 20 random scans (usually used for training)
    #   * 12 figure scans (usually used for testing)

    func_figure = [(os.path.join('func', 'data_figure_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 13)]

    func_random = [(os.path.join('func', 'data_random_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 21)]

    # Labels, 10x10 patches, stimuli shown to the subject:
    #   * 20 random labels
    #   * 12 figure labels (letters and shapes)

    label_filename = 'data_%s_run%02d_label.csv'
    label_figure = [(os.path.join('label', label_filename % ('figure', i)),
                     url, opts) for i in range(1, 13)]

    label_random = [(os.path.join('label', label_filename % ('random', i)),
                     url, opts) for i in range(1, 21)]

    # Masks

    file_mask = [
        'mask.nii.gz',
        'LHlag0to1.nii.gz',
        'LHlag10to11.nii.gz',
        'LHlag1to2.nii.gz',
        'LHlag2to3.nii.gz',
        'LHlag3to4.nii.gz',
        'LHlag4to5.nii.gz',
        'LHlag5to6.nii.gz',
        'LHlag6to7.nii.gz',
        'LHlag7to8.nii.gz',
        'LHlag8to9.nii.gz',
        'LHlag9to10.nii.gz',
        'LHV1d.nii.gz',
        'LHV1v.nii.gz',
        'LHV2d.nii.gz',
        'LHV2v.nii.gz',
        'LHV3A.nii.gz',
        'LHV3.nii.gz',
        'LHV4v.nii.gz',
        'LHVP.nii.gz',
        'RHlag0to1.nii.gz',
        'RHlag10to11.nii.gz',
        'RHlag1to2.nii.gz',
        'RHlag2to3.nii.gz',
        'RHlag3to4.nii.gz',
        'RHlag4to5.nii.gz',
        'RHlag5to6.nii.gz',
        'RHlag6to7.nii.gz',
        'RHlag7to8.nii.gz',
        'RHlag8to9.nii.gz',
        'RHlag9to10.nii.gz',
        'RHV1d.nii.gz',
        'RHV1v.nii.gz',
        'RHV2d.nii.gz',
        'RHV2v.nii.gz',
        'RHV3A.nii.gz',
        'RHV3.nii.gz',
        'RHV4v.nii.gz',
        'RHVP.nii.gz'
    ]

    file_mask = [(os.path.join('mask', m), url, opts) for m in file_mask]

    file_names = func_figure + func_random + \
                 label_figure + label_random + \
                 file_mask

    files = _fetch_files('miyawaki2008', file_names, resume=resume,
                         data_dir=data_dir)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:])
