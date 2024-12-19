from download_with_session_ID_Dec192024 import *
import sys,os,glob,subprocess
import pandas as pd

# print("I AM HERE")
# Example usage if run directly
def call_pyradiomics(SESSION_ID,file_output_dir):
    SCAN_ID,SCAN_NAME=get_selected_scan_info(SESSION_ID,file_output_dir)
    download_an_xmlfile_with_URIString_func(SESSION_ID,f'{SESSION_ID}.xml',file_output_dir)
    project_name,subject_name, session_label,acquisition_site_xml,acquisition_datetime_xml,scanner_from_xml,body_part_xml,kvp_xml=get_info_from_xml(os.path.join(file_output_dir,f'{SESSION_ID}.xml'))
    variable_dict={"project_name":project_name,"subject_name":subject_name, "session_label":session_label,"acquisition_site_xml":acquisition_site_xml,"acquisition_datetime_xml":acquisition_datetime_xml,"scanner_from_xml":scanner_from_xml,"body_part_xml":body_part_xml,"kvp_xml":kvp_xml}
    df1 = pd.DataFrame([variable_dict])
    df1['session_id']=SESSION_ID
    df1['scan_id']=SCAN_ID
    df1['scan_name']=SCAN_NAME
    df1.to_csv(os.path.join(file_output_dir,'radiomics.csv'),index=False)
    print(SCAN_ID)
    return 1
if __name__ == "__main__":
    # import sys
    try:
        SESSION_ID=sys.argv[1]
        file_output_dir=sys.argv[2]
        call_pyradiomics(SESSION_ID,file_output_dir)
        print(f"I AM HERE::{sys.argv[3:]}")
    except Exception as e:
        print(f"I FAILED::{e}")

    #
    # # Ensure input arguments are provided
    # if len(sys.argv) != 3:
    #     print("Usage: python run_pyradiomics.py <grayscale_image> <mask_image>")
    #     sys.exit(1)
    #
    # gray_image = sys.argv[1]
    # mask_image = sys.argv[2]
    #
    # try:
    #     output_file = extract_radiomics_features(gray_image, mask_image)
    #     print(f"Radiomics features saved to {output_file}")
    # except Exception as e:
    #     print(str(e))
    #     sys.exit(1)
