import re
from file_preprocessing import preprocess



def findCryptoLine(java_pth, apk_decom):
    pre_java_pth = ''
    
    rule_criteria = [
        {"import": ("javax.crypto.*", "javax.crypto.Cipher"), "regex": ("Cipher.getInstance\s*\(",)}, # Rule1
        {"import": ("javax.crypto.spec.*", "javax.crypto.spec.SecretKeySpec"), "regex": ("new\s*SecretKeySpec\s*\(",)}, # Rule2
        {"import": ("java.security.*", "java.security.KeyStore"), "regex":()}, # Rule3
        {"import": ("java.security.*", "java.security.KeyPairGenerator"), "regex":()}, # Rule4
        {"import": ("javax.crypto.spec.*", "javax.crypto.spec.IvParameterSpec"), "regex": ("new\s*IvParameterSpec\s*\(",)}, # Rule5
    ]
            
    with open(java_pth, "r") as f:
        line_dic = {}
        code = f.read()
    
    
    for c_ in range(len(rule_criteria)):
        rule = rule_criteria[c_]
        if any(imp in code for imp in rule["import"]):
            if c_ == 2:
                var_list = set()
                var_list.update(re.findall(r'KeyStore\s+(\w+)\s*;', code))
                var_list.update(re.findall(r'KeyStore\s+(\w+)\s*=', code))
                
                if var_list:
                    for var in var_list:
                        rule["regex"] = rule["regex"] + (f"{var}.load\s*\(",)
                        rule["regex"] = rule["regex"] + (f"{var}.store\s*\(",)
                        rule["regex"] = rule["regex"] + (f"{var}.getKey\s*\(",)

            if c_ == 3:
                var_list = set()
                var_list.update(re.findall(r'KeyPairGenerator\s+(\w+)\s*;', code))
                var_list.update(re.findall(r'KeyPairGenerator\s+(\w+)\s*=', code))
                
                if var_list:
                    for var in var_list:
                        rule["regex"] = rule["regex"] + (f"{var}.initialize\s*\(",)
       
       
            for regex in rule["regex"]:
                if re.search(regex, code):
                    pre_java_pth = preprocess(java_pth, apk_decom)
                    with open(pre_java_pth, "r") as f:
                        code_list = f.readlines()
                        
                    for n in range(len(code_list)):
                        line = code_list[n]
                        for regex in rule["regex"]:
                            if re.search(regex, line):
                                if c_ == 2:
                                    if 'load' in regex:
                                        rule_num = 31
                                    elif 'store' in regex:
                                        rule_num = 32
                                    else:
                                        rule_num = 33
                                else:
                                    rule_num = c_+1
                                line_dic[n+1] = rule_num
                            else:
                                pass
                else:
                    pass
                        
    sort_dic = {key: line_dic[key] for key in sorted(line_dic.keys())}
    return sort_dic, pre_java_pth
    
    
    
