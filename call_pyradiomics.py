from download_with_session_ID_Dec192024 import *
from utilities_simple_trimmed import *
import sys,os,glob,subprocess
import pandas as pd
from fill_local_mysql_Dec262024 import *
from run_pyradiomics import *
# print("I AM HERE")
# Example usage if run directly

# def fill_local_mysql(session_id, session_name, scan_id, scan_name,column_name,column_value):
#     insert_data(session_id, session_name, scan_id, scan_name)
#     update_or_create_column(session_id, scan_id, column_name, column_value,session_name,scan_name)
#     # update_or_create_column(session_id, scan_id, 'session_name', session_name,session_name,scan_name)
#     # update_or_create_column(session_id, scan_id, 'scan_name', scan_name,session_name,scan_name)
def call_pyradiomics(SESSION_ID,file_output_dir,mask_dir_and_ext):
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
    resource_dir=mask_dir_and_ext[0]
    downloadfile_withasuffix(SESSION_ID,SCAN_ID,file_output_dir,'NIFTI','.nii')
    original_nifti=os.path.join('/input',SCAN_NAME.split('.nii')[0]+'.nii')
    for each_ext in mask_dir_and_ext[1:]:
        downloadfile_withasuffix(SESSION_ID,SCAN_ID,file_output_dir,resource_dir,each_ext)
        levelset2originalRF_new_flip_with_params(os.path.join('/input',SCAN_NAME.split('.nii')[0]+'.nii'), os.path.join('/input',SCAN_NAME.split('.nii')[0]+each_ext), '/workingoutput') #, mask_color=(0, 255, 0), image_prefix="original_ct_with_infarct_only", threshold=0.5)
    for each_ext in mask_dir_and_ext[1:]:
        this_mask=glob.glob('/workingoutput/*'+each_ext) #[0]
        for this_mask_each in this_mask:
            extract_radiomics_features(os.path.join('/input',SCAN_NAME.split('.nii')[0]+'.nii'), this_mask_each, output_csv=this_mask_each.split('.nii')[0]+'_radiomics.csv')
        # downloadfile_withasuffix(SESSION_ID,SCAN_ID,file_output_dir,resource_dir,each_ext)
        # levelset2originalRF_new_flip_with_params(os.path.join('/input',SCAN_NAME.split('.nii')[0]+'.nii'), os.path.join('/input',SCAN_NAME.split('.nii')[0]+each_ext), '/workingoutput') #, mask_color=(0, 255, 0), image_prefix="original_ct_with_infarct_only", threshold=0.5)
    # for each_radiomic_file in glob.glob(os.path.dirname(this_mask)+'/*_radiomics.csv'):
    # for each_ext in mask_dir_and_ext[1:]:
    #     this_mask=glob.glob('/workingoutput/*'+each_ext) #[0]
    #     for this_mask_each in this_mask:
    for each_radiomic_file in glob.glob(os.path.dirname(this_mask_each)+'/*_radiomics.csv'):
        try:
            print("Before UPLOAD SUCCESS")
            resource_dirname='RADIOMICS'
            url='/data/experiments/'+SESSION_ID+'/scans/'+SCAN_ID ##+'/resources/'+resource_dirname
            uploadsinglefile_with_URI(url,each_radiomic_file,resource_dirname)
            print("UPLOAD SUCCESS")
        except Exception as e:
            print(e)

    #
    #         insert_data(SESSION_ID, session_label, SCAN_ID, SCAN_NAME)
    #         print('success')
    #     except:
    #         pass


    print (each_ext)
    return 1
if __name__ == "__main__":
    # import sys
    try:
        SESSION_ID=sys.argv[1]
        file_output_dir=sys.argv[2]
        call_pyradiomics(SESSION_ID,file_output_dir,sys.argv[4:])
        print(f"I AM HERE::{sys.argv[4:]}")

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
