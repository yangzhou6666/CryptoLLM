import os
import argparse
import subprocess
import time
import sys
import shutil
import torch
from tqdm import tqdm
from datetime import datetime
from slicing import slicingCode
from extract_crypto_line import findCryptoLine

from detect_codebert import model_setting_codebert, detect_codebert
from detect_codegpt import model_setting_codegpt, detect_codegpt
from detect_electra import model_setting_electra, detect_electra
from detect_codet5 import model_setting_codet5, detect_codet5


jadx_pth = 'jadx/build/jadx/bin/jadx'

def execute_command(cmd_):
    proc = subprocess.Popen(cmd_, shell=True, bufsize=256, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        proc.communicate(timeout=60)
        proc.terminate()
        proc.wait()
        
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait()
        time.sleep(10)
    
    return proc.poll()


def result(ap, args, java_pth=None, preds=[]):
    with open(f"{args.o}/{ap}.txt", "a") as f:
        if (java_pth == None) and (len(preds) == 0):
            f.write(f"Benign")
        elif (java_pth != None) and (len(preds) == 0):
            pth = java_pth.split('sources/')[1]
            f.write(f"{pth} -> X\n")
        else:
            preds = preds.tolist()
            if 1 in preds:
                r_ = 'm'
            else:
                r_ = 'b'
            pth = java_pth.split('sources/')[1]
            f.write(f"{pth} -> {r_}\n")


def main():
    parser = argparse.ArgumentParser(description='CryptoLLM')
    parser.add_argument('--f', '--folder', type=str, required=True, help='Target APK File Folder Path')
    parser.add_argument('--o', '--output', type=str, required=True, help='Ouput Folder Path')
    parser.add_argument('--p', '--model_path', type=str, required=True, help='Path of Trained Model')
    parser.add_argument('--m', '--model', type=str, required=True, help='Model to Use')
    
    args = parser.parse_args()

    if not os.path.exists(args.f):
       print("*** Folder does not exist. Put correct path.")
       sys.exit()
    
    if args.m not in ['codebert', 'codegpt', 'codet5', 'electra']:
       print("*** Model to use is wrong. Put 'codebert' or 'codegpt' or 'codet5' or 'electra'.")
       sys.exit()     

    if not os.path.exists(args.p):
       print("*** Path of trained model does not exist. Put correct path.")
       sys.exit()
                 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if args.m == 'codebert':
        model, tokenizer = model_setting_codebert(args)
    elif args.m == 'codegpt':
        model, tokenizer = model_setting_codegpt(args)
    elif args.m == 'codet5':
        model, tokenizer = model_setting_codet5(args)
    else:
        model, tokenizer = model_setting_electra(args)
            
    
    decom_pth = './decompile'    
    if not os.path.exists(decom_pth):
        os.makedirs(decom_pth) 
        
    if not os.path.exists(args.o):
        os.makedirs(args.o)
        
    print("====================================================================================================")
    print(f"** Start: {datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}")
    
    
    for ap in tqdm(os.listdir(args.f)):
        apk_pth = os.path.join(args.f, ap)
        if os.path.isfile(apk_pth):
            apk_decom = os.path.join(decom_pth, ap)
            os.makedirs(apk_decom)
            jadx_cmd = f'{jadx_pth} -d {apk_decom} {apk_pth} -r'
            done = execute_command(jadx_cmd)
            
            if done == 0:
                check = False
                for root, dirs, files in os.walk(apk_decom):
                    for file in files:
                        java_pth = os.path.join(root, file)
                        try:
                            line_dic, pre_java_pth = findCryptoLine(java_pth, apk_decom)
                            if line_dic:
                                code_nz_dir = slicingCode(file, apk_decom, pre_java_pth, line_dic, device)
                                if code_nz_dir != False:
                                    if args.m == 'codebert':
                                        preds = detect_codebert(code_nz_dir, model, tokenizer, args)
                                    elif args.m == 'codegpt':
                                        preds = detect_codegpt(code_nz_dir, model, tokenizer, args)
                                    elif args.m == 'codet5':
                                        preds = detect_codet5(code_nz_dir, model, tokenizer, args)
                                    else:
                                        preds = detect_electra(code_nz_dir, model, tokenizer, args)
                                    result(ap, args, java_pth, preds)
                                    check = True
                                else:
                                    result(ap, args, java_pth)
                        except:
                            result(ap, args, java_pth)  
                    
                if check == False:
                    result(ap, args)
            
                if os.path.exists(apk_decom):
                    shutil.rmtree(apk_decom)      
            
            else:
                with open(f"{args.o}/{ap}.txt", "w") as f:
                    f.write("X")   
    
    if os.path.exists('./graph'):
        shutil.rmtree('./graph')     


if __name__ == '__main__':
    main()