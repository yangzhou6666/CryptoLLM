import re
    

def remove_empty(f_code):
    code_list = []
    
    for c_ in f_code:
        if c_.strip()== '\n':
            pass
        else:
            code_list.append(c_.strip()+'\n')
    return code_list
          
    
def removeJADXwarn(code_list):    
    code = ''
    turn = False
    for c_ in code_list:
        if all(char in [' ', '\n'] for char in c_):
            continue
        if c_.startswith('@'):
            continue
        if '/* bridge */ /* synthetic */' in c_:
            c_ = c_.replace('/* bridge */ /* synthetic */','')
        if '/*' in c_:
            if turn == False:
                if 'replace' in c_:
                    code += c_.split('/*')[0]
                else:
                    code += re.sub('(?<![:/])//.*', '', c_.split('/*')[0])
            turn = True
        if '*/' in c_:
            if turn == True:
                if 'replace' in c_:
                    code += c_.split('*/')[1]
                else:
                    code += re.sub('(?<![:/])//.*', '', c_.split('*/')[1])
            turn = False
            continue
                
        if turn == False:
            if 'replace' in c_:
                code += c_
            else:
                code += re.sub('(?<![:/])//.*', '', c_)
    
    code_l = code.split("\n")
    code = [line + '\n' for line in code_l if line]
    return code


def else_catch(code_list):
    code = ''.join(code_list)
    code = code.replace(' {', '{')        
    code = code.replace('else{\n','else{ ')
    
    start_indexes = [m.start() for m in re.finditer('catch', code)]
    end_indexes = [m.start() for m in re.finditer(r'\n', code)]
    
    tmp = ''

    for cnt in range(len(start_indexes)):
        start_index = start_indexes[cnt]
        for end_index in end_indexes:
            if end_index > start_index:
                if len(start_indexes) == 1:
                    tmp += code[:start_index + len("catch")] + code[start_index + len("catch"):end_index+1].strip().replace("\n", "") + code[end_index+1:]
                    break
                if cnt < len(start_indexes)-1:
                    if cnt == 0:
                        tmp += code[:start_index + len("catch")] + code[start_index + len("catch"):end_index+1].strip().replace("\n", "") + code[end_index+1:start_indexes[cnt+1]]
                        break
                    else:
                        tmp += code[start_index:end_index+1].strip().replace("\n", "") + code[end_index+1:start_indexes[cnt+1]]                            
                        break
                else:
                    tmp += code[start_index:end_index+1].strip().replace("\n", "") + code[end_index+1:]
                    break
                
    if tmp != '':
        code = tmp.replace('\n}\n', '}\n')
    code = code.replace('\n}\n', '}\n')
    code = code.replace('\n}\n', '}\n')
    code = code.replace('\n}\n', '}\n')
    code = code.replace('\n}\n', '}\n')
    return code


def replace_within_quotes(code_list, max_length=30):
    pattern = r'"([^"]*)"'
    tmp = []
    for code in code_list:
        matches = re.findall(pattern, code)

        for match in matches:
            if len(match) >= max_length:
                replacement = '"' + match[:max_length] + '"'
                code = code.replace('"' + match + '"', replacement)
        
        tmp.append(code)

    return tmp   


def removefun(code_list):
    tmp = []
    for c_ in code_list:
        match0 = re.findall(r'^[a-zA-Z]{2}\.[a-zA-Z]\(\);$', c_.strip())
        match1 = re.findall(r'^[a-zA-Z]{2}\.[a-zA-Z]\(\d\);$', c_.strip())
        if not (match0 or match1):
            tmp.append(c_)
    
    return tmp
            
            
def preprocess(java_pth, apk_decom):
    with open(java_pth, "r") as f:
        f_code = f.readlines()
    code_list = remove_empty(f_code)
    code_list = replace_within_quotes(code_list, 30)
    code_list = removefun(code_list)
    code = removeJADXwarn(code_list)
    code = else_catch(code)
    
    fn = java_pth.split('/')[-1]
    pre_java_pth = f'{apk_decom}/{fn}'
    with open(pre_java_pth, "w") as f:
        f.write(code)
    
    return pre_java_pth 