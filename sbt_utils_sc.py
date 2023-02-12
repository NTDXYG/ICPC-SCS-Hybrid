import re

import execjs
import javalang

ctx = execjs.compile("""
const parser = require('@solidity-parser/parser');
function getAst(code){
    const input = `
    contract smartcontract {` +
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

def VLR(a):
    tmp_list = []
    key_list = []
    if isinstance(a,dict):
        key_list = a.keys()
    for key in key_list:
        if key == 'type':
            tmp_list.append(a['type'])
        elif isinstance(a[key],dict):
            tmp_list.extend(VLR(a[key]))
        elif isinstance(a[key],list):
            for k in a[key]:
                tmp_list.extend(VLR(k))
    return tmp_list

def hump2underline(hunp_str):
    '''
    驼峰形式字符串转成下划线形式
    :param hunp_str: 驼峰形式字符串
    :return: 字母全小写的下划线形式字符串
    '''
    # 匹配正则，匹配小写字母和大写字母的分界位置
    p = re.compile(r'([a-z]|\d)([A-Z])')
    # 这里第二个参数使用了正则分组的后向引用
    sub = re.sub(p, r'\1 \2', hunp_str).lower()
    return sub

def process_source(code):
    code = code.replace('\n',' ').strip()
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        tks = []
        for tk in tokens:
            if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                tks.append('STR_')
            elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                tks.append('NUM_')
            elif tk.__class__.__name__ == 'Boolean':
                tks.append('BOOL_')
            else:
                tks.append(hump2underline(tk.value))
    except Exception:
        code = code.replace("\r","")
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        result_list = re.split(pattern, code)
        tks = [hump2underline(t) for t in result_list]
    return " ".join(tks)

def transformer(code):
    try:
        ast = ctx.call("getAst", code)  # 调用js函数add，并传入它的参数
        sbt = ' '.join(VLR(ast))
    except:
        sbt = 'SourceUnit'
    return process_source(code), sbt

code = """
      function balanceOf(address addr) constant public returns (uint256) {
      	return data.balanceOf(addr);
      }
    """

print(transformer(code))