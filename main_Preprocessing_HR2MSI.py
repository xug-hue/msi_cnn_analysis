import math
import numpy as np
import csv
from pathlib import Path
import pyimzml.ImzMLParser as imzmlp

RESULT_PATH = str(Path(__file__).parent / 'MSI/result/HR2MSI')

# HR2MSI mouse urinary bladder S096 data can be downloaded at https://www.ebi.ac.uk/pride/archive/projects/PXD001283
IMZML_PATH = str(Path(__file__).parent / 'MSI/data/HR2MSI mouse urinary bladder S096.imzML')
IBD_PATH = str(Path(__file__).parent / 'MSI/data/HR2MSI mouse urinary bladder S096.ibd')
PARSE_LIB = ['lxml']
TEST_CASES = [(parse_lib, IMZML_PATH, IBD_PATH)
              for parse_lib in PARSE_LIB]
MZS_LOW_VALUE = 400.0
MZS_HIGH_VALUE = 1000.0
PPM = 10

def normalize(ints):
    sumint = np.sum(ints)
    return ints / sumint * 1000000

def calibrate_byPPM(mzs, ints):
    method = "local_sum"
    mzStandardValue = MZS_LOW_VALUE
    mzerror = mzStandardValue * PPM / 1000000
    decimal = int(math.log(1 / mzerror, 10)) + 1
    mzsStandard = np.array([MZS_LOW_VALUE, ])
    intsStandard = np.array([0.0, ])
    for i in range(len(mzs)):
        while mzStandardValue < mzs[i]-np.around(mzerror, decimal)/2:
            mzStandardValue += np.around(mzerror, decimal)
            mzerror = mzStandardValue * PPM / 1000000
            decimal = int(math.log(1 / mzerror, 10)) + 1
            mzsStandard = np.append(mzsStandard, [mzStandardValue])
            intsStandard = np.append(intsStandard, [0])
        if (abs(mzStandardValue - mzs[i]) <= np.around(mzerror, decimal) / 2):
            if (method == "local_sum"):
                intsStandard[len(intsStandard) - 1] += ints[i]
    while mzStandardValue < MZS_HIGH_VALUE - np.around(mzerror, decimal):
        mzStandardValue += np.around(mzerror, decimal)
        mzerror = mzStandardValue * PPM / 1000000
        decimal = int(math.log(1 / mzerror, 10)) + 1
        mzsStandard = np.append(mzsStandard, [mzStandardValue])
        intsStandard = np.append(intsStandard, [0])
    return mzsStandard, intsStandard

for parse_lib, imzml_path, ibd_path in TEST_CASES:
    with imzmlp.ImzMLParser(imzml_path, parse_lib=parse_lib) as parser:
        md = parser.metadata
        if len(md.scan_settings) == 1:
            pixelNumX = md.scan_settings['scansettings1']['max count of pixels x']
            pixelNumY = md.scan_settings['scansettings1']['max count of pixels y']
        minlen = 10000
        maxlen = 0
        intsumNParray = np.zeros((pixelNumY, pixelNumX))
        for pixelIndex in range(pixelNumX*pixelNumY):
            mzs, ints = parser.getspectrum(pixelIndex)
            assert len(mzs) == len(ints)
            if len(mzs) < minlen:
                minlen = len(mzs)
            if len(mzs) > maxlen:
                maxlen = len(mzs)
            # save raw data
            with open('MSI/data/raw'+str(pixelIndex)+'.csv', 'w+', encoding='utf-8', newline="") as f:
                for mznum in range(len(mzs)):
                    csv_write = csv.writer(f)
                    csv_write.writerow([mzs[mznum], ints[mznum]])
            # calibration
            mzs, ints = calibrate_byPPM(mzs, ints)
            # normalization
            ints = normalize(ints)
            path = RESULT_PATH + '\\' + parse_lib + str(pixelIndex) + '(' + str(pixelIndex // pixelNumX) + ',' + str(pixelIndex % pixelNumX) + ')' + '.csv'
            # save normalization result in separated csv
            with open(path, 'w+', encoding='utf-8', newline="") as f:
                for mznum in range(len(mzs)):
                    csv_write = csv.writer(f)
                    csv_write.writerow([ints[mznum]])
            # intsum nparray
            intsumNParray[pixelIndex//pixelNumX][pixelIndex % pixelNumX] = np.sum(ints)/1000000