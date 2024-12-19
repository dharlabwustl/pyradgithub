
import os, sys, errno, shutil, uuid,subprocess,csv,json
import math,inspect
import glob
import re,time
import requests
import pandas as pd
import nibabel as nib
import numpy as np
# import pydicom as dicom
import pathlib
import argparse,xmltodict
from xnatSession import XnatSession
# from redcapapi_functions import *
catalogXmlRegex = re.compile(r'.*\.xml$')
XNAT_HOST_URL=os.environ['XNAT_HOST']  #'http://snipr02.nrg.wustl.edu:8080' #'https://snipr02.nrg.wustl.edu' #'https://snipr.wustl.edu'
XNAT_HOST = XNAT_HOST_URL # os.environ['XNAT_HOST'] #
XNAT_USER = os.environ['XNAT_USER']#
XNAT_PASS =os.environ['XNAT_PASS'] #
api_token=os.environ['REDCAP_API']
xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
class arguments:
    def __init__(self,stuff=[]):
        self.stuff=stuff
def download_a_singlefile_with_URIString(url,filename,dir_to_save):
    print("url::{}::filename::{}::dir_to_save::{}".format(url,filename,dir_to_save))
    # xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()
    # command="echo  " + url['URI'] + " >> " +  os.path.join(dir_to_save,"test.csv")
    # subprocess.call(command,shell=True)
    response = xnatSession.httpsess.get(xnatSession.host +url) #/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv") #
    #                                                       # "/data/experiments/SNIPR02_E03548/scans/1-CT1/resources/147851/files/ICH_0001_01022017_0414_1-CT1_threshold-1024.0_22121.0TOTAL_VersionDate-11302022_04_22_2023.csv") ## url['URI'])
    zipfilename=os.path.join(dir_to_save,filename ) #"/data/projects/ICH/resources/179772/files/ICH_CTSESSIONS_202305170753.csv")) #sessionId+scanId+'.zip'
    with open(zipfilename, "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    # xnatSession.close_httpsession()
    return zipfilename

def get_selected_scan_info(SESSION_ID, dir_to_save):
    # Log failure message for debugging
    command = f"echo failed at each_row_id: {inspect.stack()[0][3]} >> output/error.txt"
    subprocess.call(command, shell=True)

    # Define the URI and URL for the request
    URI = f'/data/experiments/{SESSION_ID}'
    url = f"{URI}/resources/NIFTI_LOCATION/files?format=json"

    # Initialize XNAT session
    xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()

    try:
        # Make a GET request to the XNAT server
        response = xnatSession.httpsess.get(xnatSession.host + url)
        response.raise_for_status()  # Raise an error for unsuccessful responses
        metadata_masks = response.json()['ResultSet']['Result']

        # Convert metadata to a DataFrame
        df_scan = pd.read_json(json.dumps(metadata_masks))

        # Log additional debug information
        command = f"echo failed_1 at each_row_id: {df_scan['URI'].iloc[0]} >> output/error.txt"
        subprocess.call(command, shell=True)

        # Download the file specified by the URI
        download_a_singlefile_with_URIString(
            str(df_scan['URI'].iloc[0]),
            os.path.basename(str(df_scan['URI'].iloc[0])),
            dir_to_save
        )

        # Read the downloaded file to extract SCAN_ID and SCAN_NAME
        nifti_location = pd.read_csv(
            os.path.join(dir_to_save, os.path.basename(str(df_scan['URI'].iloc[0])))
        )
        SCAN_ID = str(nifti_location.loc[nifti_location.index[0], 'ID'])
        SCAN_NAME = str(nifti_location.loc[nifti_location.index[0], 'Name'])

        # Log success message
        command = f"echo passed at each_row_id: {inspect.stack()[0][3]} >> output/error.txt"
        subprocess.call(command, shell=True)

        return SCAN_ID, SCAN_NAME

    except Exception as e:
        # Log exception details for debugging
        command = f"echo exception at {inspect.stack()[0][3]}: {str(e)} >> output/error.txt"
        subprocess.call(command, shell=True)
        raise e

    finally:
        # Close the XNAT session
        xnatSession.close_httpsession()


def get_metadata_session(sessionId,outputfile="NONE.csv"):
    url = ("/data/experiments/%s/scans/?format=json" %    (sessionId))
    xnatSession = XnatSession(username=XNAT_USER, password=XNAT_PASS, host=XNAT_HOST)
    xnatSession.renew_httpsession()
    response = xnatSession.httpsess.get(xnatSession.host + url)
    xnatSession.close_httpsession()
    metadata_session=response.json()['ResultSet']['Result']
    metadata_session_1=json.dumps(metadata_session)
    df_scan = pd.read_json(metadata_session_1)
    df_scan.to_csv(outputfile,index=False)
    return metadata_session