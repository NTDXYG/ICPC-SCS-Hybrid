import json
import random
import re

import execjs
import javalang

ctx = execjs.compile("""
const parser = require('@solidity-parser/parser');
function getAst(code){
    const input = `
    contract SmartContract {` +
        code
            + `
    }`
    try {
        var ast = parser.parse(input)
        return ast
    } catch (e) {
        if (e instanceof parser.ParserError) {
            console.error(e.errors)
            var ast = {"type": "SourceUnit"}
            return ast
        }
    }
}
""") # 获取代码编译完成后的对象


def VLR(a, level=0):
    tmp_list = []
    key_list = []
    if isinstance(a,dict):
        key_list = a.keys()
    for key in key_list:
        if key == 'type':
            if('name' in key_list):
                d = {'level': level, 'type': a['type'], 'value': a['name']}
            else:
                d = {'level': level, 'type': a['type'], 'value': 'Null'}
            tmp_list.append(d)
        elif isinstance(a[key],dict):
            tmp_list.extend(VLR(a[key], level = level+1))
        elif isinstance(a[key],list):
            for k in a[key]:
                tmp_list.extend(VLR(k, level = level+1))
    return tmp_list


def get_ast(code):
    result = ctx.call("getAst", code) # 调用js函数add，并传入它的参数
    ast_temp = VLR(result)
    ast_new = []
    for i in range(len(ast_temp)):
        d = ast_temp[i]
        d['id'] = i
        if ('value' not in d.keys()):
            d['value'] = 'Null'
        ast_new.append(d)

    level_list = []
    for i in range(len(ast_new)):
        level_list.append(ast_new[i]['level'])

    for i in range(len(ast_new)):
        level = ast_new[i]['level']
        if (level < max(level_list)):
            child_list = []
            for j in range(i + 1, len(ast_new)):
                if (level != ast_new[j]['level']):
                    if (ast_new[j]['level'] == level + 1):
                        child_list.append(j)
                else:
                    break
            if (len(child_list) > 0):
                ast_new[i]['children'] = child_list
    return ast_new


def AST_(cur_root_id, node_list):
    color = ['B','R','Y','G','DI']
    i = random.randint(1,4)
    cur_root = node_list[cur_root_id]
    result_list = []
    tmp_dict = {}
    tmp_dict['id'] = str(cur_root['id'])
    tmp_dict['variableType'] = cur_root['type']
    tmp_dict['name'] = 'Null'
    if (cur_root_id == 0):
        tmp_dict['status'] = color[0]
    else:
        tmp_dict['status'] = color[i]

    if 'children' in cur_root:
        chs = cur_root['children']
        temp_list = []
        for ch in chs:
            temp_node = AST_(ch, node_list)
            if(isinstance(temp_node, list)):
                temp_list.extend(temp_node)
            else:
                temp_list.append(temp_node)
        tmp_dict['children'] = temp_list

    if 'value' in cur_root:
        tmp_dict['name'] = cur_root['value']
        if tmp_dict['name'] == None:
            tmp_dict['name'] = 'Null'
        return tmp_dict

    result_list.append(tmp_dict)
    return result_list


def parse_ast(code):
    ast = get_ast(code)
    ast = json.dumps(ast)
    ast = json.loads(ast)
    ast_tree = AST_(0, ast)
    ast_tree = str(ast_tree).replace('\'', '\"')
    return ast_tree

code = """
function balanceOf(address addr) constant public returns (uint256) {
    return data.balanceOf(addr);
}
    """

