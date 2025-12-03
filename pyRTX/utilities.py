import numpy as np
import spiceypy as sp
import pandas as pd
from datetime import datetime

def export_formatted(accelerations, epochs, filename, units = None):
    """
    Export acceleration data to formatted text file with epoch timestamps.

    Writes acceleration vectors and their corresponding epochs to a CSV file
    with descriptive header. The output format is suitable for reading by
    orbit determination software or data analysis tools.

    Parameters
    ----------
    accelerations : ndarray, shape (N, 3)
        Array of 3-component acceleration vectors. Each row is [ax, ay, az]
        at one time step. Must be 2D array with exactly 3 columns.
    epochs : ndarray, shape (N,)
        Array of epoch times corresponding to each acceleration vector.
        Should be in seconds past J2000 (SPICE ephemeris time format).
    filename : str
        Path to output file. Typically uses .txt or .csv extension.
    units : str or None, default=None
        Optional string describing the units of the acceleration data
        (e.g., 'km/s^2', 'm/s^2'). Will be added to file header if provided.

    Returns
    -------
    None
        Data is written to file at specified filename.

    Raises
    ------
    ValueError
        If accelerations is not shape (N, 3) or if the number of epochs
        doesn't match the number of acceleration vectors.

    Notes
    -----
    Output file format:
    - CSV format with comma delimiters
    - Header line with format description and optional units
    - Data columns: Epoch, X, Y, Z
    - One row per time step

    The file header contains:
    - Description of data format
    - Column definitions (Epoch in seconds past J2000, X, Y, Z components)
    - Units information if provided

    Example output file:
    ```
    # Acceleration file. Columns: Epoch (seconds past J2000), X, Y, Z. Units: km/s^2
    0.000000000000000000e+00,1.234567890123456789e-08,2.345678901234567890e-08,3.456789012345678901e-08
    6.000000000000000000e+01,1.234567890123456789e-08,2.345678901234567890e-08,3.456789012345678901e-08
    ...
    ```

    The output can be read back using:
    ```python
    data = np.loadtxt(filename, delimiter=',')
    epochs = data[:, 0]
    accelerations = data[:, 1:4]
    ```

    Examples
    --------
    >>> epochs = np.array([0.0, 60.0, 120.0])
    >>> accels = np.array([[1e-8, 2e-8, 3e-8],
    ...                    [1.1e-8, 2.1e-8, 3.1e-8],
    ...                    [1.2e-8, 2.2e-8, 3.2e-8]])
    >>> export_formatted(accels, epochs, 'output.txt', units='km/s^2')

    See Also
    --------
    exportEXAC : Export to GEODYN EXAC binary format
    numpy.savetxt : Underlying function for text output
    """
    if not len(accelerations.shape) == 2 or not accelerations.shape[1] == 3:
        raise ValueError('Error, the acceleration vector must be (N,3)')
    
    if accelerations.shape[0] != len(epochs):
        raise ValueError(f'Error the number of epochs ({len(epochs)}) is different from the number of accelerations ({accelerations.shape[0]})')
    todump = np.vstack([epochs.T, accelerations.T]).T
    toadd = ''
    if units != None:
        toadd = f'Units: {units}'
    
    np.savetxt(filename, todump, delimiter = ',', header = 'Acceleration file. Columns: Epoch (seconds past J2000, X, Y, Z). ' + toadd)
    
    
def export_exac(satelliteID, data, tstep, startTime, endTime, outFileName):
    """
    Export acceleration data to GEODYN EXAC (External Accelerations) file format.
    
    GEODYN is NASA's precision orbit determination software. The EXAC format is
    a Fortran-formatted binary file used to provide time-varying external
    accelerations (such as solar radiation pressure) to the orbit propagator.
    
    Parameters
    ----------
    satelliteID : int
        Satellite identifier code used in GEODYN processing.
    data : ndarray, shape (N, 3)
        Acceleration data to be written, in km/s². Each row is [ax, ay, az]
        at one time step.
    tstep : int or float
        Time step between data records in seconds (e.g., 60 for 1-minute data).
    startTime : datetime.datetime
        Start time of the data series. Must include date and time information
        down to microseconds.
    endTime : datetime.datetime
        End time of the data series. Must be consistent with len(data) and tstep.
    outFileName : str
        Path to output EXAC file. Typically uses .exac or .bin extension.
    
    Returns
    -------
    None
        Data is written to binary file at outFileName.
    
    Notes
    -----
    EXAC File Structure:
    - Master header record: Control parameters and file type identifier
    - Satellite-specific header: Satellite ID, time step, start/end times
    - Data records: Time stamp + 3D acceleration vector + padding zeros
    
    Time Format:
    - Stored as YYMMDDHHMMSSμμμμμμ (year-month-day-hour-minute-second-microsecond)
    - Year uses 2-digit format (YY)
    
    Coordinate System:
    - Accelerations should be in the same reference frame as the GEODYN
      orbit integration (typically J2000 or ICRF)
    
    Units:
    - Accelerations: km/s²
    - Time step: seconds
    
    This format is used for high-precision orbit determination where external
    non-gravitational forces (solar pressure, atmospheric drag, etc.) need to
    be accurately modeled.
    
    Requires scipy.io.FortranFile for binary I/O operations.
    """

    from scipy.io import FortranFile
    import datetime

    satid = satelliteID
    date0 = startTime
    date1 = endTime
    dt = tstep
    deltatime = datetime.timedelta(seconds=dt)
    outfile = FortranFile(outFileName, 'w')

    # General Header
    masterhdr = np.array([-6666666.0, 1, 1, 0, 0, 0, 0, 0, 0])
    outfile.write_record(masterhdr)

    # Satellite specific header
    sathdr = np.array([-7777777.0, 1, satid, dt, float(date0.strftime('%Y%m%d%H%M%S')[2:]), float(date0.strftime('%f')),
                       float(date1.strftime('%Y%m%d%H%M%S')[2:]), float(date1.strftime('%f')), 0])
    outfile.write_record(sathdr)

    # Data records
    date = date0 - deltatime
    for d_elem in data:
        date = date + deltatime
        datarec = np.array(
            [float(date.strftime('%Y%m%d%H%M%S')[2:]), float(date.strftime('%f')), d_elem[0], d_elem[1], d_elem[2], 0,
             0, 0, 0])
        outfile.write_record(datarec)
    
    

def to_datetime(epoch):
    t = sp.et2utc(epoch, 'C', 6)
    dt = datetime.strptime(t, "%Y %b %d %H:%M:%S.%f")
    return dt
    
    
    

def getScPosVel(spacecraft, center, epochs, frame):
    correction = 'CN'
    sc_pos, _ = sp.spkezr(spacecraft, epochs, frame, correction, center)
    sc_pos = np.array(sc_pos)
    # Returns position and velocity
    return sc_pos[:, :3], sc_pos[:, 3:]
