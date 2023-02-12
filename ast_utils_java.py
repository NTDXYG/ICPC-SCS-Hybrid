import collections
import json
import random
import re
import javalang

COMMENT_RX = re.compile("(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/", re.MULTILINE)

def parse_java(code):
    import javalang
    from javalang.ast import Node
    def get_token(node):
        token = ''
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
    p = re.compile(r'([a-z]|\d)([A-Z])')
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


def get_sbt(code):
    code = COMMENT_RX.sub('', code)
    processed_code = process_source(code)
    sbt = parse_java(processed_code)
    code_seq = ' '.join([hump2underline(i) for i in processed_code.split()])
    return code_seq, sbt


def parse_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})
    ign = False
    outputs = []
    stop = False
    for i, Node in enumerate(flatten):
        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        children = []
        for child in node.children:
            child_path = None
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        d["id"] = i
        n = str(node)
        n = n[:n.find('(')]
        d["type"] = n
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            print(type(node))
            print(code)
            ign = True
            # break
        outputs.append(d)
    if not ign:
        return json.dumps(outputs)

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

def get_ast(code):
    code = COMMENT_RX.sub('', code)
    processed_code = process_source(code)
    ast = parse_ast(processed_code)
    ast = json.loads(ast)
    ast_tree = AST_(0, ast)
    ast_tree = str(ast_tree).replace('\'', '\"')
    return ast_tree


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

print(get_ast(code))
print(parse_ast(code))
print(get_sbt(code))