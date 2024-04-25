import subprocess
import os
import pydotplus
import re
import shutil
import signal



def execute_command(cmd_):
    proc = subprocess.Popen(cmd_, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    
    try:
        proc.communicate(timeout=120)
        proc.wait()
        if proc.returncode == 0:
            return True
        else:
            return False
        
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        return False

                
                
def makeGraph(file, snippet_pth):
    output_dot_file = './output.dot'
    output_png_file = './output.png'
    
    dot_pth = './graph/ast/dot'
    png_pth = './graph/ast/png'
    if not os.path.exists(dot_pth):
        os.makedirs(dot_pth)
    if not os.path.exists(png_pth):
        os.makedirs(png_pth)  
    
    dot_file = f"{dot_pth}/{file}.dot"
    png_file = f"{png_pth}/{file}.png"
            
    comex_cmd = f'comex --lang "java" --code-file {snippet_pth} --graphs ast'
    done = execute_command(comex_cmd)
    
    if done == True:
        os.rename(output_dot_file, dot_file)
        os.rename(output_png_file, png_file)
        return dot_file
    else:
        return False



def extractFeature(dot_pth):
    target_word = []
    excep_target_word = []

    graph = pydotplus.graph_from_dot_file(dot_pth)

    for node in graph.get_node_list():
        node_num = node.get_name()
        if node_num.isdigit():
            if node_num == '1':
                break
        
        node_type = node.get_attributes().get('node_type')
        if node_type == 'identifier' or node_type == 'this':
            target = node.get_attributes().get('label')
            if target == 'Exception' or target == 'new':
                continue
            if ('StringUtil\.' in target) or ('AESUtils\.' in target):
                continue
            if '.' in target:
                if 'this\.' in target:
                    target = target.replace('"','').split('.')[-1]
                else:
                    target = target.replace('"','').replace('\\','')
            if target not in target_word:
                target_word.append(target)            
        
        if node_type == 'type_identifier':
            excep = node.get_attributes().get('label')
            if excep not in excep_target_word:
                excep_target_word.append(excep)
                
    target_word = [x for x in target_word if x not in excep_target_word]
    return target_word


    
def normalization(file, apk_decom, snippet_pth, line_num):    
    code_nz_dir = f'{apk_decom}/code_snippet_nz/{file}'
    if not os.path.exists(code_nz_dir):
        os.makedirs(code_nz_dir)
        
    code_nz_pth = f'{code_nz_dir}/{file}_{line_num}.java'
            
    dot_pth = makeGraph(file, snippet_pth)
    
    if dot_pth == False:
        shutil.copy(snippet_pth, code_nz_pth)
    else:
        target_word = extractFeature(dot_pth)
        if 'java.util.Locale.ENGLISH' in target_word:
            target_word.remove('java.util.Locale.ENGLISH')
        if 'Math' in target_word:
            target_word.remove('Math')
        if 'Integer.MAX_VALUE' in target_word:
            target_word.remove('Integer.MAX_VALUE')
        if 'Integer' in target_word:
            target_word.remove('Integer')
        if 'System' in target_word:
            target_word.remove('System')
        if 'Base64' in target_word:
            target_word.remove('Base64')
        
        with open(snippet_pth, "r") as f:
            code = f.readlines()
            code_str = ''.join(code)
            
# ============================= FUN normalization =============================
        nz_fun_list = []
        fnum = 0
        for word in target_word:
            if re.search(r'\b{}\b\s*\('.format(re.escape(word)), code_str):
                nz_fun_list.append(word)            
        fun_patterns = [re.compile(r"\b{}\b\s*\(".format(re.escape(fun))) for fun in nz_fun_list]
        
        for c_ in code:
            if c_.startswith(('public', 'private', 'protected', 'void')) and '{' in c_:
                for p_ in fun_patterns:
                    if re.search(p_, c_):
                        fnum += 1
                        code_str = re.sub(p_, f'FUN{fnum}(', code_str)

        
# ============================= VAR normalization =============================
        nz_var_list = []
        for word in target_word:
            if re.search(r'\b{}\b\s*=\s*'.format(re.escape(word)), code_str) \
                or re.search(r'\b{}\b\s*[,]'.format(re.escape(word)), code_str) \
                or re.search(r'\b{}\b\s*[)]'.format(re.escape(word)), code_str) \
                or re.search(r'[(]\s*\b{}\b'.format(re.escape(word)), code_str) \
                or re.search(r'\b{}\b\s*[.]'.format(re.escape(word)), code_str) \
                or re.search(r'\b{}\b\s*;\s*'.format(re.escape(word)), code_str):
                    nz_var_list.append(word)

        var_patterns = [re.compile(r'(?<!")\b{}\b(?!["(])'.format(re.escape(var))) for var in nz_var_list]
  
        for num in range(len(var_patterns)):
            code_str = re.sub(var_patterns[num], f'VAR{num+1}', code_str)
        
        with open(code_nz_pth, 'a') as f:
            f.write(''.join(code_str))
            
    return code_nz_dir