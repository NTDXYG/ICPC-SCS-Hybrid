import re

import javalang

COMMENT_RX = re.compile("(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/", re.MULTILINE)

def parse_java(code):
    import javalang
    from javalang.ast import Node

    def get_token(node):
        token = ''
        # if isinstance(node, str):
        #     token = node.replace('\"|\'', '')
        # elif isinstance(node, set):
        #     token = 'Modifier'  # node.pop()
        # elif
        if isinstance(node, Node):
            token = node.__class__.__name__

        return token

    def get_children(root):
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def get_sequence(node, sequence):
        token, children = get_token(node), get_children(node)
        sequence.append(token)
        for child in children:
            get_sequence(child, sequence)

    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    seq = []
    get_sequence(tree, seq)
    result = ' '.join(seq)
    return ' '.join(result.split())

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


def process_source(code):
    code = code.replace('\n',' ').strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('_STR')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('_NUM')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('_BOOL')
        else:
            tks.append(tk.value)
    return " ".join(tks)

code = """
        public static Optional<Node> search(final Node node, final String name) {
        if (node.getName().equals(name)) {
            return Optional.of(node);
        }

        return node.getSubNodes()
                .stream()
                .map(value -> search(value, name))
                .flatMap(Optional::stream)
                .findAny();
    }
    """

def transformer(code):
    code = COMMENT_RX.sub('', code)
    code_seq = ' '.join([hump2underline(i) for i in process_source(code).split()])
    sbt = parse_java(code)
    return code_seq, sbt

print(transformer(code))