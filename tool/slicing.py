import os
import subprocess
import pydotplus
import pandas as pd
import re
import signal
from normalization import normalization



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
    
    
    
def changeFormatDotFile(output_dot_file, dot_file):
    with open(output_dot_file, "r") as input_f, open(dot_file, "w") as output_f:
        for line in input_f:
            if 'label=' in line:
                line = line.replace('\\','')
                split_line = line.split('"')
                for i in range(1,len(split_line)):
                    if i==1 or i==len(split_line)-1:
                        split_line[i] = '"'+split_line[i]
                    else:
                        split_line[i] = "'"+split_line[i]
                output_f.write(''.join(split_line))
                
            elif '->' in line:
                split_line = line.split('"')
                for i in range(1,len(split_line)):
                    if i==1 or i==len(split_line)-1:
                        split_line[i] = '"'+split_line[i]
                    else:
                        split_line[i] = "'"+split_line[i]
                output_f.write(''.join(split_line))
            
            else:
                output_f.write(line)
       
                

def makeGraph(file, java_pth, type):
    output_dot_file = './output.dot'
    output_png_file = './output.png'
    dot_pth = './graph/dot'
    png_pth = './graph/png'
    
    if not os.path.exists(dot_pth):
        os.makedirs(dot_pth)
    if not os.path.exists(png_pth):
        os.makedirs(png_pth)    
    
    file = file.replace('/','_')
    dot_file = f'{dot_pth}/{file}.dot'
    png_file = f'{png_pth}/{file}.dot'
            
    comex_cmd = f'comex --lang "java" --code-file {java_pth} --graphs {type}'
    done = execute_command(comex_cmd)
    
    if done == True:
        changeFormatDotFile(output_dot_file, dot_file)
        os.remove(output_dot_file)
        os.rename(output_png_file, png_file)
        return dot_file
    else:
        print(f"[Error in slicing] Cannot make graph from : {java_pth}")
        return False
    
    

def splitDot(file, dot_file):
    cfg_pth = './graph/cfg'
    dfg_pth = './graph/dfg'
    
    if not os.path.exists(cfg_pth):
        os.makedirs(cfg_pth)
    if not os.path.exists(dfg_pth):
        os.makedirs(dfg_pth) 
    
    file = file.replace('/','_')
    cfg_file = f'graph/cfg/{file}.dot'
    dfg_file = f'graph/dfg/{file}.dot'
    
    with open(dot_file, "r") as input_f, open(cfg_file, "w") as cfg_f, open(dfg_file, "w") as dfg_f:
        for line in input_f:
            if '->' in line:
                if 'color=red' in line:
                    cfg_f.write(line)
                else:
                    dfg_f.write(line)
            else:
                cfg_f.write(line)
                dfg_f.write(line)
                
    return cfg_file, dfg_file



def extractFeature(dot_pth, java_pth):
    graph = pydotplus.graph_from_dot_file(dot_pth)
    graph_feature = []
    
    with open(java_pth, "r") as f:
        java_code = f.readlines()

    for node in graph.get_node_list():
        node_num = node.get_name()
        if node_num.isdigit():
            if node_num == '1':
                break
        
        label = node.get_attributes().get('label').replace('"', '')
        line_num = label.split('_',1)[0]
        code = java_code[int(line_num)-1]
        type = node.get_attributes().get('type_label')
        graph_feature.append((node_num, line_num, code, type))

    graph_df = pd.DataFrame(graph_feature, columns=['node', 'line', 'code', 'type'])
    return graph_df



def findVAR(code, rule_num):
    var_list = []
    
    if rule_num == 1:
        api_fun = [r'getInstance\s*\(']
    elif rule_num == 2:
        api_fun = [r'new\s*SecretKeySpec\s*\(']
    elif rule_num == 31:
        api_fun = [r'load\s*\(']
    elif rule_num == 32:
        api_fun = [r'store\s*\(']
    elif rule_num == 33:
        api_fun = [r'getKey\s*\(']
    elif rule_num == 4:
        api_fun = [r'initialize\s*\(']
    else:
        api_fun = [r'new\s*IvParameterSpec\s*\(']
    
    for api in api_fun:
        match = re.search(api, code)
        if match is None:
            pass
        else:
            try:
                target = code[match.end():]
                t_count, bracket = 0, 1
                for t_ in list(target):
                    t_count += 1
                    if '(' == t_:
                        bracket += 1
                    elif ')' == t_:
                        bracket -= 1
                    if bracket == 0:
                        break
        
                params = ''.join(list(target)[0:t_count-1])
                
                if 'java.util.Locale.ENGLISH' in params:
                    var_list.append(params)

                else:
                    comma, brackets, p_count = 0, 0, 0
                    if ',' in params:
                        for p_ in list(params):
                            p_count += 1
                            if p_ in '([{':
                                brackets += 1
                            elif p_ in ')]}':
                                brackets -= 1
                            elif p_ == ',':
                                comma += 1
                                if comma > 0 and brackets == 0:
                                    if rule_num in [1,2,4,5]:
                                        params = ''.join(list(params)[0:p_count-1]).strip()
                                    else:
                                        params = ''.join(list(params)[p_count:]).strip()
                                    break
                            else:
                                continue 
                    else:
                        pass
                
                    if params.startswith('this.'):
                        this = params.split('this.')[1]
                        if '.' in this:
                            var_list.append(f"\"{this}\"")
                        else:
                            var_list.append(this)
                    else:                        
                        p_list = params.split('.')
                        for p in p_list:
                            if (p == 'this') or ('()' in p):
                                pass
                            elif ('(' and ')' in p):
                                var_list.append(p.split('(')[1].split(')')[0])
                            else:
                                var_list.append(p)
            except:
               pass 
    return var_list  



def dfgBW(target_node, dfg_pth, var_list, graph_df):
    graph = pydotplus.graph_from_dot_file(dfg_pth)
    bw_dfg = {str(target_node)}
    src_list = {str(target_node)}
    visited_nodes = set()
    
    first = True
    while src_list:
        new_nodes = set()
        for src in src_list:
            visited_nodes.add(src)
            for edge in graph.get_edge_list():
                start = edge.get_source()
                end = edge.get_destination()
                if end == str(src):
                    if (start not in visited_nodes) and (start != '1'):
                        used_def = edge.get("used_def")
                        if first == True:
                            if (used_def == None) or (used_def in var_list):
                                new_nodes.add(start)
                        else:
                            new_nodes.add(start)
                            
            src_list = new_nodes
            bw_dfg.update(src_list)
            
            first = False
    bw_dfg = sorted(bw_dfg, key=int)
    return bw_dfg 
            


def cfgBW(bw_dfg, cfg_pth, graph_df):
    graph = pydotplus.graph_from_dot_file(cfg_pth)
    bw_cfg = set(bw_dfg)
    visited_nodes = set()
       
    for bw in bw_dfg:
        src_list = {str(bw)}
        while src_list:
            new_nodes = set()
            for src in src_list:
                visited_nodes.add(src)
                for edge in graph.get_edge_list():
                    start = edge.get_source()
                    end = edge.get_destination()
                    if (end == str(src)) and (start != '1'):
                        code = graph_df.loc[graph_df['node'] == start, 'code'].values[0]
                        type = graph_df.loc[graph_df['node'] == end, 'type'].values[0]
                        label = edge.get("label")
                        if (label == 'method_return' and start not in visited_nodes) or \
                            (start in visited_nodes) or \
                            (code.startswith('import')) or \
                            (label == 'class_next' and label == 'construct_next'):
                                continue
                        else:
                            new_nodes.add(start)
                            
                src_list = new_nodes
                bw_cfg.update(src_list)
    bw_cfg = sorted(bw_cfg, key=int)
    return bw_cfg     



def cfgFW(bw_dfg, cfg_pth, graph_df):
    graph = pydotplus.graph_from_dot_file(cfg_pth)
    fw_cfg = set(bw_dfg)
    visited_nodes = set()
    
    for bw in bw_dfg:
        src_list = {str(bw)}
        while src_list:
            new_nodes = set()
            for src in src_list:
                visited_nodes.add(src)
                for edge in graph.get_edge_list():
                    start = edge.get_source()
                    end = edge.get_destination()
                    if (start == str(src)) and (end != '1') :
                        code = graph_df.loc[graph_df['node'] == end, 'code'].values[0]
                        label = edge.get("label")
                        if (label == 'method_return' and end not in visited_nodes) or \
                            (end in visited_nodes) or \
                            (code.startswith('import')) or \
                            (label == 'construct_next' and label== 'class_next'):
                                continue
                        else:
                            new_nodes.add(end)
                          
                src_list = new_nodes
                fw_cfg.update(src_list)
    fw_cfg = sorted(fw_cfg, key=int)
    return fw_cfg   



def ifdfgBW(if_node, dfg_pth):
    graph = pydotplus.graph_from_dot_file(dfg_pth)
    if_dfg = set()
    
    for edge in graph.get_edge_list():
        start = edge.get_source()
        end = edge.get_destination()
        if end == str(if_node):
            if (start != '1'):
                if_dfg.add(start)
                        
    if_dfg = sorted(if_dfg, key=int)
    return if_dfg     
    


def toCode(snippet_node, graph_df, dfg_pth):
    snippet_code = []
    for node in snippet_node:
        if node != '1':
            code = graph_df.loc[graph_df['node'] == node, 'code'].values[0]
            if code.strip().startswith('if'):
                if_dfg = ifdfgBW(node, dfg_pth)
                for if_ in if_dfg:
                    if_code = graph_df.loc[graph_df['node'] == if_, 'code'].values[0]
                    if if_code not in snippet_code:
                        snippet_code.append(if_code)
            if code not in snippet_code:
                snippet_code.append(code)
        
    return snippet_code
    
    
    
def slicingCode(file, apk_decom, java_pth, line_dic, device):
    snippet_dir = f'{apk_decom}/code_snippet/{file}'

    if not os.path.exists(snippet_dir):
        os.makedirs(snippet_dir)
    
    dot_pth = makeGraph(file, java_pth, 'cfg,dfg')
    
    if dot_pth == False:
        return False
    
    else:
        graph_df = extractFeature(dot_pth, java_pth)
        cfg_pth, dfg_pth = splitDot(file, dot_pth)
        
        for line_num, rule_num in line_dic.items():
            code = graph_df.loc[graph_df['line'] == str(line_num), 'code'].values[0]
            var_list = findVAR(code, rule_num)
            
            target_node = graph_df.loc[graph_df['line'] == str(line_num), 'node'].values[0]
            bw_dfg = dfgBW(target_node, dfg_pth, var_list, graph_df)
            bw_cfg = cfgBW(bw_dfg, cfg_pth, graph_df)
            fw_cfg = cfgFW(bw_dfg, cfg_pth, graph_df)
                        
            snippet_node = list(set(bw_dfg + bw_cfg + fw_cfg))
            snippet_node = sorted(snippet_node, key=int)  
            snippet_code = ''.join(toCode(snippet_node, graph_df, dfg_pth))
            
            file = file.replace('/','_')
            snippet_pth = f'{snippet_dir}/{file}_{line_num}.java'
            with open(snippet_pth, 'w') as f:
                f.write(snippet_code)
                
            code_nz_pth = normalization(file, apk_decom, snippet_pth, line_num)

            
    return code_nz_pth        
    
    