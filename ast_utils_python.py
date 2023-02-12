import ast, re
import json
import random

def parse_python(code):

    def str_node(node):
        if isinstance(node, ast.AST):
            rv = '%s' % (node.__class__.__name__)
            return rv

    def ast_visit(node, level=0):
        d = {'level':level, 'type':str_node(node)}
        if isinstance(node, str):
            d['value'] = node
        tokens.append(d)

        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        ast_visit(item, level=level + 1)
            elif isinstance(value, ast.AST) and not isinstance(value, ast.Load) and not isinstance(value, ast.Store) and not isinstance(value, str):
                ast_visit(value, level=level + 1)
            elif isinstance(value, str):
                d['value'] = value
        return tokens
    tokens = []
    try:
        tree = ast.parse(code)
        ast_visit(tree)
        ast_temp = tokens
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
    except SyntaxError:
        return 'Error'


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
    ast = parse_python(code)
    ast = json.dumps(ast)
    ast = json.loads(ast)
    ast_tree = AST_(0, ast)
    ast_tree = str(ast_tree).replace('\'', '\"')
    return ast_tree

code = """
def circumference ( r ) :
    return ( 2 * PI * r )
"""

print(parse_ast(code))