import re
from sctokenizer import PythonTokenizer, TokenType
COMMENT_RX = re.compile("(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/", re.MULTILINE)

def python_tokenize(line):
    tokenizer = PythonTokenizer()
    tokens = tokenizer.tokenize(line)
    l = []
    for token in tokens:
        if (token.token_type == TokenType.STRING):
            l.append('STR_')
        elif (token.token_type == TokenType.CONSTANT):
            l.append('NUM_')
        elif (token.token_value.find('_') != -1):
            token.token_value = token.token_value.split('_')
            l.extend(token.token_value)
        # elif (token.token_value != "'"):
        else:
            l.append(token.token_value)
    # return [i for i in l if i != "'"]
    return l

def parse_python(code):
    import ast,re, asttokens
    def str_node(node):
        if isinstance(node, ast.AST):
            rv = '%s' % (node.__class__.__name__)
            return rv
        else:
            return repr(node)

    def ast_visit(node, level=0):
        tokens.extend(str_node(node).split())
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST) and not isinstance(item, ast.Load) and not isinstance(item, ast.Store):
                        ast_visit(item, level=level + 1)
            elif isinstance(value, ast.AST) and not isinstance(value, ast.Load) and not isinstance(value, ast.Store):
                ast_visit(value, level=level + 1)
        return tokens
    tokens = []
    try:
        tree = ast.parse(code)
        ast_visit(tree)
        result = ' '.join(tokens)
        return result
    except SyntaxError:
        return 'Error'

def getCode(raw_code):
    srcLine = python_tokenize(raw_code)
    srcLine = [hump2underline(s) for s in srcLine]
    return ' '.join(srcLine)

def hump2underline(hunp_str):
    '''
    驼峰形式字符串转成下划线形式
    :param hunp_str: 驼峰形式字符串
    :return: 字母全小写的下划线形式字符串
    '''
    # 匹配正则，匹配小写字母和大写字母的分界位置
    p = re.compile(r'([a-z]|\d)([A-Z])')
    # 这里第二个参数使用了正则分组的后向引用
    sub = re.sub(p, r'\1 \2', hunp_str)
    return sub

def transformer(code):
    code = COMMENT_RX.sub('', code)
    code_seq = getCode(code)
    sbt = parse_python(code)
    return code_seq, sbt

code = """
def get_svc_avail_path ( ): 
    print("hello")
    return 5 
"""
print(transformer(code))
